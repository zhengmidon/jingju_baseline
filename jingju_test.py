"""
  This script provides an exmaple to wrap UER-py for generation.
  Given the beginning of a text, language model generates the rest.
"""
import sys
import os
import argparse
import torch
import torch.nn.functional as F
import time
from tqdm import tqdm
import pickle

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.layers import *
from uer.encoders import *
from uer.targets import *
from uer.utils.constants import *
from uer.utils import *
from uer.utils.nlp_metrics import *
from uer.utils.config import load_hyperparam
from uer.model_loader import load_model
from uer.opts import infer_opts, tokenizer_opts


class GenerateLm(torch.nn.Module):
    def __init__(self, args):
        super(GenerateLm, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.target = str2target[args.target](args, len(args.tokenizer.vocab))

    def forward(self, src, tgt, seg):
        emb = self.embedding(src, seg)
        e_output = self.encoder(emb, seg)
        output = self.target.output_layer(e_output)
        loss_info = self.target(e_output, tgt)
        return output, loss_info


def top_k_top_p_filtering(logits, top_k, top_p):
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float("Inf")

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float("Inf")
    return logits

def read_beginning(args):
    with open(args.beginning_path, mode="r", encoding="utf-8") as f:
        src_tensor = []
        seg_tensor = []
        line = []
        target_lengths = []
        beginning_length = []
        lines = f.readlines()
        print(f'there are {len(lines)} instances')
        for i in range(len(lines)):
            line.append(lines[i].split('\t')[0])
            target_lengths.append(lines[i].split('\t')[1])
            #src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(line[i]))
            src = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(line[i]))

            seg = [1] * len(src)
            beginning_length.append(len(src))
            if len(src) > args.seq_length:
                src = src[:args.seq_length]
                seg = seg[:args.seq_length]
            src_tensor.append(src)
            seg_tensor.append(seg)
        print(f'length of src:{len(src_tensor[0])},{len(src_tensor[1])},{len(src_tensor[2])}')
        print(f'length of seg:{len(seg_tensor[0])},{len(seg_tensor[1])},{len(seg_tensor[2])}')
        print(f'example----\n{src_tensor[0]}\n{line[0]}')
    assert len(src_tensor) == len(target_lengths)
    '''
    for i in range(len(target_lengths)):
        print(f'{i}\t{target_lengths[i]}')
    '''
    return (src_tensor, seg_tensor, beginning_length, target_lengths)
    #src_tensor, seg_tensor = torch.LongTensor(src_tensor), torch.LongTensor(seg_tensor)

def generate(args,model,device,inputs,n_iter):
    src_tensor,seg_tensor, beginning_length, target_lengths = inputs
    with open(args.prediction_path, mode="w", encoding="utf-8") as f:
        src_tensor_list = []
        seg_tensor_list = []
        n = len(src_tensor) if n_iter >= len(src_tensor) else n_iter
        for k in range(n):
            star_time = time.time()
            target_temp = torch.LongTensor([src_tensor[k]])
            generate_target = target_temp
            seg_temp = torch.LongTensor([seg_tensor[k]])
            for i in range(int(target_lengths[k]) - beginning_length[k]):
                with torch.no_grad():
                    output, loss_info = model(target_temp.to(device), seg_temp.to(device), seg_temp.to(device))
                output = output.cpu()
                next_token_logits = output[0][-1] / args.temperature
                filtered_logits = top_k_top_p_filtering(next_token_logits, args.top_k, args.top_p)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

                if len(target_temp[0]) == 1024:
                    target_temp = torch.cat([target_temp[:,1:], next_token.view(1,1)], dim=-1)
                    #seg_temp = torch.cat([seg_temp[:,1:], torch.tensor([[1]])], dim=-1)

                else:
                    target_temp = torch.cat([target_temp, next_token.view(1,1)], dim=-1)
                    seg_temp = torch.cat([seg_temp, torch.tensor([[1]])], dim=-1)
                generate_target = torch.cat([generate_target, next_token.view(1,1)], dim=-1)
                '''
                if next_token.item() == 102:
                    break
                '''
            src_tensor_list.append(generate_target)
            #print(f'length of src_list:{len(src_tensor_list[k][0])}')

            #f.write(line[k] + "\n")
            generated_sentence = "".join(
                args.tokenizer.convert_ids_to_tokens(
                [token_id.item() for token_id in src_tensor_list[k][0][beginning_length[k]:]])
            )
            f.write(generated_sentence + '\n')
            end_time = time.time()
            print(f"time consumption in {k}th loop:{end_time - star_time:.2f} seconds")

def prepare_beginning_and_reference(args):
        print('------preparing beginning and reference------')
        begin = time.time()
        testset_reader = open(args.testset_path, "rb")
        begging_file = open(args.beginning_path, "w")
        reference_file = open(args.reference_path, "w")
        while True:
            try:
                instance_info = pickle.load(testset_reader)
                instance = torch.tensor(instance_info[0])
                instance = instance[instance > 0]
                target_length = instance_info[1]

                beginning = instance[:100]
                reference = instance[100:]

                beginning_sentence = "".join(
                args.tokenizer.convert_ids_to_tokens([token_id.item() for token_id in beginning])
                )
                reference_sentence = "".join(
                args.tokenizer.convert_ids_to_tokens([token_id.item() for token_id in reference])
                )

                begging_file.write(beginning_sentence + '\t' + str(target_length) + '\n')
                reference_file.write(reference_sentence + '\n')

            except EOFError:
                break
        print('------end preparing ------')
        testset_reader.close()
        begging_file.close()
        reference_file.close()

def cal_bleu_f1_distinct(args):
    with open(args.reference_path,'r') as ref:
        references = ref.readlines()
    with open(args.prediction_path,'r') as candi:
        candidates = candi.readlines()
    n_pairs = min(len(references),len(candidates))
    references = references[:n_pairs]
    candidates = candidates[:n_pairs]
    #assert len(references) == len(candidates)
    ref_list = []
    candi_list = []
    for i in range(len(references)):
        ref_list.append(args.tokenizer.tokenize(references[i]))
        candi_list.append(args.tokenizer.tokenize(candidates[i]))
    #print(f'ref example{ref_list[0]},candi example{candi_list[0]}') 
    pairs = [list(i) for i in zip(candi_list, ref_list)]
    # calc f1
    f1 = calc_f1(pairs)
    # calc bleu
    bleu1, bleu2 = calc_bleu(pairs)
    # calc distinct
    distinct1, distinct2 = calc_distinct(pairs)

    output_str = "F1: %.2f%%\n" % (f1 * 100)
    output_str += "BLEU1: %.3f%%\n" % (bleu1*100)
    output_str += "BLEU2: %.3f%%\n" % (bleu2*100)
    output_str += "DISTINCT1: %.3f%%\n" % (distinct1*100)
    output_str += "DISTINCT2: %.3f%%\n" % (distinct2*100)
    print('='*20)
    sys.stdout.write(output_str)


def cal_ppl(args,model,device):
    print('------testing------')
    begin = time.time()
    testset_reader = open(args.testset_path, "rb")
    total_loss = 0.
    n_batches = 0
    while True:
        instances = []
        try:
            for i in range(args.batch_size):
                instances.append(pickle.load(testset_reader))

            src = []
            tgt = []
            seg = []

            for ins in instances:
                src.append(ins[0][:-1])
                tgt.append(ins[0][1:])
                if ins[1] == len(ins[0]):
                    seg.append([1] * (ins[1] - 1))
                else:
                    seg.append([1] * ins[1] + [0] * (len(ins[0]) - 1 - ins[1]))


            src = torch.LongTensor(src).to(device)
            tgt = torch.LongTensor(tgt).to(device)
            seg = torch.LongTensor(seg).to(device)
            with torch.no_grad():
                output, loss_info = model(src, tgt, seg)
            loss, correct, denominator = loss_info
            total_loss += loss.item()

            n_batches += 1

        except EOFError:
                break
    #model.train()
    testset_reader.close()
    avg_loss = total_loss / n_batches
    avg_loss = round(avg_loss, 4)
    print('='*20)
    print(f'Test perplexity:{math.exp(avg_loss):.2f}')
    end = time.time()
    print(f'time usage {end-begin}s\n------test complete------')


if __name__ == '__main__':
    device = torch.device("cuda")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    infer_opts(parser)

    parser.add_argument("--target", choices=["lm"], default="lm",
                        help="The training target of the pretraining model.")
    parser.add_argument("--has_lmtarget_bias", action="store_true",
                        help="Add bias on output_layer for lm target.")
    parser.add_argument("--tie_weights", action="store_true",
                        help="Tie the word embedding and softmax weights.")
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--testset_path", type=str, required=True,
                        help="Path to the test set.")
    parser.add_argument("--beginning_path", type=str, required=True,
                        help="Path to the test beginnings.")
    parser.add_argument("--reference_path", type=str, required=True,
                        help="Path to the test references.")

    tokenizer_opts(parser)

    args = parser.parse_args()

    args.batch_size = 5

    args = load_hyperparam(args)

    args.tokenizer = str2tokenizer[args.tokenizer](args)

    model = GenerateLm(args)
    model = load_model(model, args.load_model_path)
    model = model.to(device)
    model.eval()

    if (not os.path.exists(args.beginning_path)) or (not os.path.exists(args.reference_path)):
        print(f'{args.beginning_path} and {args.reference_path}do not exist,create!')
        prepare_beginning_and_reference(args)
    #prepare_beginning_and_reference(args)
    #inputs = read_beginning(args)
    if not os.path.exists(args.prediction_path):
        print(f'{args.prediction_path} does not exist,create it!')
        inputs = read_beginning(args)
        generate(args,model,device,inputs,1000)
    cal_bleu_f1_distinct(args)
    cal_ppl(args,model,device)
