#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This code is based on the tutorial by Sean Robertson <https://github.com/spro/practical-pytorch> found here:
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

Students *MAY NOT* view the above tutorial or use it as a reference in any way.
"""

from __future__ import unicode_literals, print_function, division

import argparse
import logging
import random
import time
from io import open

import matplotlib

# if you are running on the gradx/ugradx/ another cluster,
# you will need the following line
# if you run on a local machine, you can comment it out
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu
from torch import optim
import math
from torch.autograd import Variable
import numpy as np

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

# we are forcing the use of cpu, if you have access to a gpu, you can set the flag to "cuda"
# make sure you are very careful if you are using a gpu on a shared cluster/grid,
# it can be very easy to confict with other people's jobs.
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = "<SOS>"
EOS_token = "<EOS>"
PAD_token = "<PAD>"

SOS_index = 0
EOS_index = 1
PAD_index = 2
MAX_LENGTH = 15
BATCH_SIZE = 3


class Vocab:
    """ This class handles the mapping between the words and their indicies
    """

    def __init__(self, lang_code):
        self.lang_code = lang_code
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_index: SOS_token, EOS_index: EOS_token, PAD_index: PAD_token}
        self.n_words = 3  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self._add_word(word)

    def _add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


######################################################################


def split_lines(input_file):
    """split a file like:
    first src sentence|||first tgt sentence
    second src sentence|||second tgt sentence
    into a list of things like
    [("first src sentence", "first tgt sentence"),
     ("second src sentence", "second tgt sentence")]
    """
    logging.info("Reading lines of %s...", input_file)
    # Read the file and split into lines
    lines = open(input_file, encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs
    pairs = [l.split('|||') for l in lines]
    return pairs


def make_vocabs(src_lang_code, tgt_lang_code, train_file):
    """ Creates the vocabs for each of the langues based on the training corpus.
    """
    src_vocab = Vocab(src_lang_code)
    tgt_vocab = Vocab(tgt_lang_code)

    train_pairs = split_lines(train_file)

    for pair in train_pairs:
        src_vocab.add_sentence(pair[0])
        tgt_vocab.add_sentence(pair[1])

    logging.info('%s (src) vocab size: %s', src_vocab.lang_code, src_vocab.n_words)
    logging.info('%s (tgt) vocab size: %s', tgt_vocab.lang_code, tgt_vocab.n_words)

    return src_vocab, tgt_vocab


######################################################################

def tensor_from_sentence(vocab, sentence):
    """creates a tensor from a raw sentence
    """
    indexes = []
    for word in sentence.split():
        try:
            indexes.append(vocab.word2index[word])
        except KeyError:
            pass
            # logging.warn('skipping unknown subword %s. Joint BPE can produces subwords at test time which are not in vocab. As long as this doesnt happen every sentence, this is fine.', word)
    indexes.append(EOS_index)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_from_pair(src_vocab, tgt_vocab, pair):
    """creates a tensor from a raw sentence pair
    """
    input_tensor = tensor_from_sentence(src_vocab, pair[0])
    target_tensor = tensor_from_sentence(tgt_vocab, pair[1])
    return input_tensor, target_tensor


######################################################################


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, dropout=0.6):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.w_i = torch.nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.w_f = torch.nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.w_o = torch.nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.w_c = torch.nn.Parameter(torch.Tensor(hidden_size, input_size))

        self.u_i = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.u_f = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.u_o = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.u_c = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        self.bias1 = torch.nn.Parameter(torch.Tensor(hidden_size).uniform_())
        self.bias2 = torch.nn.Parameter(torch.Tensor(hidden_size).uniform_())
        self.bias3 = torch.nn.Parameter(torch.Tensor(hidden_size).uniform_())
        self.bias4 = torch.nn.Parameter(torch.Tensor(hidden_size).uniform_())


        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hidden, pre_cell):

        input = input.view(input.size()[0], -1)
        hidden = hidden.view(hidden.size()[0], -1)
        pre_cell = pre_cell.view(pre_cell.size()[0], -1)

        f_t = self.sigmoid(torch.mm(input, self.w_f) + torch.mm(hidden, self.u_f) + self.bias1)
        i_t = self.sigmoid(torch.mm(input, self.w_i) + torch.mm(hidden, self.u_i) + self.bias2)
        o_t = self.sigmoid(torch.mm(input, self.w_o) + torch.mm(hidden, self.u_o) + self.bias3)

        c_t = self.tanh(torch.mm(input, self.w_c) + torch.mm(hidden, self.u_c) + self.bias4)
        c_t = torch.mul(pre_cell, f_t) + torch.mul(i_t, c_t)
        h_t = torch.mul(o_t, self.tanh(c_t))
        h_t = h_t.view(h_t.size()[0], 1, -1)
        c_t = c_t.view(c_t.size()[0], 1, -1)
        F.dropout(h_t, p=self.dropout, inplace=True)
        return h_t, c_t


class EncoderRNN(nn.Module):
    """the class for the enoder RNN
    """

    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        """Initilize a word embedding and bi-directional LSTM encoder
        For this assignment, you should *NOT* use nn.LSTM. 
        Instead, you should implement the equations yourself.
        See, for example, https://en.wikipedia.org/wiki/Long_short-term_memory#LSTM_with_a_forget_gate
        You should make your LSTM modular and re-use it in the Decoder.
        """
        "*** YOUR CODE HERE ***"
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, bidirectional=True)

    def forward(self, input_batch, hidden, input_lengths, batched=False):
        """runs the forward pass of the encoder
        returns the output and the hidden state
        """
        "*** YOUR CODE HERE ***"
        #print(input_batch.shape, self.input_size, self.hidden_size)
        if batched:
            embedded = self.embedding(input_batch)
            outputs = embedded
            packed = torch.nn.utils.rnn.pack_padded_sequence(outputs, input_lengths)
            cn = self.get_batched_hidden_state()
            outputs, (hidden, cn) = self.lstm(packed, (hidden, cn))

            outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
            return outputs, hidden
        else:
            outputs = torch.zeros(MAX_LENGTH, 2 * self.hidden_size, device=device)
            input_length = input_batch.size(0)

            pre_cell = self.get_initial_hidden_state()

            for ei in range(input_length):
                input = input_batch[ei]
                embedded = self.embedding(input).view(1, 1, -1)
                output = embedded
                output, (hidden, pre_cell) = self.lstm(output, (hidden, pre_cell))
                outputs[ei][:self.hidden_size] = hidden[0, 0]
            pre_cell = self.get_initial_hidden_state()
            hidden = self.get_initial_hidden_state()
            for ei in reversed(range(input_length)):
                input = input_batch[ei]
                embedded = self.embedding(input).view(1, 1, -1)
                output = embedded
                output, (hidden, pre_cell) = self.lstm(output, (hidden, pre_cell))

                outputs[ei][self.hidden_size:] = hidden[0, 0]
            outputs = outputs[:, :self.hidden_size] + outputs[:, self.hidden_size:]
            return outputs, hidden


        # outputs = torch.zeros(MAX_LENGTH, 2 * self.hidden_size, device=device)
        # input_length = input_batch.size(0)
        #
        # # embedded = self.embedding(input_batch).view(1, 1, -1)
        # # output = embedded
        # #
        # pre_cell = self.get_initial_hidden_state()
        # # output, (hidden, pre_cell) = self.lstm(output, (hidden, pre_cell))
        # # return output, hidden
        #
        # for ei in range(input_length):
        #     input = input_batch[ei]
        #     embedded = self.embedding(input).view(1, 1, -1)
        #     output = embedded
        #     #print(output[0].shape)
        #     output, (hidden, pre_cell) = self.lstm(output, (hidden, pre_cell))
        #     outputs[ei][:self.hidden_size] = hidden[0, 0]
        # pre_cell = self.get_initial_hidden_state()
        # hidden = self.get_initial_hidden_state()
        # for ei in reversed(range(input_length)):
        #     input = input_batch[ei]
        #     embedded = self.embedding(input).view(1, 1, -1)
        #     output = embedded
        #     output, (hidden, pre_cell) = self.lstm(output, (hidden, pre_cell))
        #
        #     outputs[ei][self.hidden_size:] = hidden[0, 0]
        # return outputs, hidden

    def get_batched_hidden_state(self):
        return torch.zeros(2, BATCH_SIZE, self.hidden_size, device=device)


    def get_initial_hidden_state(self):
        return torch.zeros(2, 1, self.hidden_size, device=device)


class Attention(nn.Module):

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.attention_v = nn.Parameter(torch.ones(self.hidden_size)) #FloatTensor(1, self.hidden_size))

   # def forward(self, hidden, outputs):


class AttnDecoderRNN(nn.Module):
    """the class for the decoder
    """

    def __init__(self, hidden_size, output_size, dropout_p=0.7, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.dropout = nn.Dropout(self.dropout_p)

        """Initilize your word embedding, decoder LSTM, and weights needed for your attention here
        """
        "*** YOUR CODE HERE ***"
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)

        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, bidirectional=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.attention = nn.Linear(self.hidden_size * 2, self.hidden_size) #max_length)
        self.attentionNB = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attention_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.attention_v = nn.Parameter(torch.ones(self.hidden_size)) #FloatTensor(1, self.hidden_size))
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, input, hidden, encoder_outputs, batched=False):
        """runs the forward pass of the decoder
        returns the log_softmax, hidden state, and attn_weights

        Dropout (self.dropout) should be applied to the word embeddings.
        """

        "*** YOUR CODE HERE ***"
        if batched:
            embedded = self.dropout(self.embedding(input).view(1, BATCH_SIZE, self.hidden_size))

            cn = self.get_batched_hidden_state()
            output, (hidden, cn) = self.lstm(embedded, (hidden, cn))
            output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]

            attn_weights = torch.tensor(torch.zeros(BATCH_SIZE, encoder_outputs.size(0)), device=device)
            for i in range(BATCH_SIZE):
                for j in range(encoder_outputs.size(0)):
                    #attn_weights[i, j] = self.attention(torch.cat((embedded[0, i].unsqueeze(0), hidden[0, i].unsqueeze(0)), 1))
                    #print(output[:, i].shape, encoder_outputs[j, i].unsqueeze(0).shape)

                    temp = self.attention(torch.cat((output[:, i], encoder_outputs[j, i].unsqueeze(0)), 1))

                    attn_weights[i, j] = self.attention_v.view(-1).dot((temp.view(-1)))

                    #attn_weights[i, j] = hidden[:, i].dot(encoder_outputs[j, i].unsqueeze(0))

            attn_weights = self.softmax(attn_weights).unsqueeze(1).view(1, BATCH_SIZE, -1)

            attn_applied = torch.bmm(attn_weights.transpose(0, 1), encoder_outputs.transpose(0, 1).squeeze(1)).transpose(0, 1).squeeze(0)
            #cont = attn_weights.bmm(encoder_outputs.transpose(0, 1)).squeeze(1)
            output = output.squeeze(0)

            output = self.tanh(self.attention_combine(torch.cat((output, attn_applied), 1))) # OR USE ReLU?
            output = self.out(output)
            return output, hidden, attn_weights

        else:
            embedded = self.dropout(self.embedding(input).view(1, 1, -1))
            attn_weights = self.softmax(self.attentionNB(torch.cat((embedded[0], hidden[0]), 1)))

            attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

            output = torch.cat((embedded[0], attn_applied[0]), 1)

            output = self.relu(self.attention_combine(output).unsqueeze(0))

            cn = self.get_initial_hidden_state()

            output, (hidden, cn) = self.lstm(output, (hidden, cn))

            log_softmax = F.log_softmax(self.out(hidden[0]), dim=1)
            return log_softmax, hidden, attn_weights


    def get_batched_hidden_state(self):
        return torch.zeros(2, BATCH_SIZE, self.hidden_size, device=device)


    def get_initial_hidden_state(self):
        return torch.zeros(2, 1, self.hidden_size, device=device)


######################################################################

def train(input_tensor, target_tensor, encoder, decoder, optimizer,
          criterion, input_lengths, target_lengths, max_length=MAX_LENGTH):
    encoder_hidden = encoder.get_batched_hidden_state()

    # make sure the encoder and decoder are in training mode so dropout is applied
    encoder.train()
    decoder.train()

    "*** YOUR CODE HERE ***"

    optimizer.zero_grad()

    target_length = target_tensor.size(0)

    loss = 0

    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden, input_lengths, batched=True)

    decoder_input = torch.tensor([SOS_index] * BATCH_SIZE, device=device)

    decoder_hidden = encoder_hidden

    max_tgt_length = max(target_lengths)

    decoder_outputs = torch.tensor(torch.zeros(max_tgt_length, BATCH_SIZE, decoder.output_size), device=device)

    #decoder_outputs = torch.tensor(torch.zeros(max_tgt_length, BATCH_SIZE, decoder.output_size), device=device)

    for i in range(max_tgt_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs, batched=True)
        #topv, topi = decoder_output.topk(1)

        #decoder_outputs[i] = topi.squeeze().detach()
        decoder_outputs[i] = decoder_output
        decoder_input = target_tensor[i]

        loss += criterion(decoder_output, target_tensor[i])

        # print(decoder_output, decoder_input)
        # if decoder_input.item() == PAD_token:
        #     break


    # for i in range(target_length):
    #     decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
    #     topv, topi = decoder_output.topk(1)
    #
    #     decoder_input = topi.squeeze().detach()
    #
    #     loss += criterion(decoder_output, target_tensor[i])
    #     if decoder_input.item() == EOS_token:
    #         break

    loss.backward()

    optimizer.step()

    return loss.item() / target_length


######################################################################

def translate(encoder, decoder, sentence, src_vocab, tgt_vocab, max_length=MAX_LENGTH):
    """
    runs tranlsation, returns the output and attention
    """

    # switch the encoder and decoder to eval mode so they are not applying dropout
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        #print(sentence)
        input_tensor = tensor_from_sentence(src_vocab, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.get_initial_hidden_state()

        input_lengths = [len(input_tensor)]

        # encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        # for ei in range(input_length):
        #     encoder_output, encoder_hidden = encoder(input_tensor[ei],
        #                                              encoder_hidden)
        #     encoder_outputs[ei] += encoder_output[0, 0]

        #encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)
        encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden, input_lengths)

        decoder_input = torch.tensor([[SOS_index]], device=device)

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_index:
                decoded_words.append(EOS_token)
                break
            else:
                decoded_words.append(tgt_vocab.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


######################################################################

# Translate (dev/test)set takes in a list of sentences and writes out their transaltes
def translate_sentences(encoder, decoder, pairs, src_vocab, tgt_vocab, max_num_sentences=None, max_length=MAX_LENGTH):
    output_sentences = []
    for pair in pairs[:max_num_sentences]:
        output_words, attentions = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab)
        output_sentence = ' '.join(output_words)
        output_sentences.append(output_sentence)
    return output_sentences


######################################################################
# We can translate random sentences  and print out the
# input, target, and output to make some subjective quality judgements:
#

def translate_random_sentence(encoder, decoder, pairs, src_vocab, tgt_vocab, n=1):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


######################################################################

def show_attention(input_sentence, output_words, attentions, fig, ax):
    """visualize the attention mechanism. And save it to a file.
    Plots should look roughly like this: https://i.stack.imgur.com/PhtQi.png
    You plots should include axis labels and a legend.
    you may want to use matplotlib.
    """

    "*** YOUR CODE HERE ***"

    ax.matshow(attentions.numpy(), cmap='gray')

    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    plt.show()



def translate_and_show_attention(input_sentence, encoder1, decoder1, src_vocab, tgt_vocab, fig, ax):
    output_words, attentions = translate(
        encoder1, decoder1, input_sentence, src_vocab, tgt_vocab)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    show_attention(input_sentence, output_words, attentions, fig, ax)


def clean(strx):
    """
    input: string with bpe, EOS
    output: list without bpe, EOS
    """
    return ' '.join(strx.replace('@@ ', '').replace(EOS_token, '').strip().split())


######################################################################

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden_size', default=256, type=int,
                    help='hidden size of encoder/decoder, also word vector size')
    ap.add_argument('--n_iters', default=10000, type=int,
                    help='total number of examples to train on')
    ap.add_argument('--print_every', default=5000, type=int,
                    help='print loss info every this many training examples')
    ap.add_argument('--checkpoint_every', default=10000, type=int,
                    help='write out checkpoint every this many training examples')
    ap.add_argument('--initial_learning_rate', default=0.001, type=int,
                    help='initial learning rate')
    ap.add_argument('--src_lang', default='fr',
                    help='Source (input) language code, e.g. "fr"')
    ap.add_argument('--tgt_lang', default='en',
                    help='Source (input) language code, e.g. "en"')
    ap.add_argument('--train_file', default='data/fren.train.bpe',
                    help='training file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence')
    ap.add_argument('--dev_file', default='data/fren.dev.bpe',
                    help='dev file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence')
    ap.add_argument('--test_file', default='data/fren.test.bpe',
                    help='test file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence' +
                         ' (for test, target is ignored)')
    ap.add_argument('--out_file', default='out.txt',
                    help='output file for test translations')
    ap.add_argument('--load_checkpoint', nargs=1,
                    help='checkpoint file to start from')
    ap.add_argument('--fig', default='output.png',
                    help='output figure for attention')

    args = ap.parse_args()

    # process the training, dev, test files

    # Create vocab from training data, or load if checkpointed
    # also set iteration
    if args.load_checkpoint is not None:
        state = torch.load(args.load_checkpoint[0])
        iter_num = state['iter_num']
        src_vocab = state['src_vocab']
        tgt_vocab = state['tgt_vocab']
    else:
        iter_num = 0
        src_vocab, tgt_vocab = make_vocabs(args.src_lang,
                                           args.tgt_lang,
                                           args.train_file)

    encoder = EncoderRNN(src_vocab.n_words, args.hidden_size).to(device)
    decoder = AttnDecoderRNN(args.hidden_size, tgt_vocab.n_words, dropout_p=0.1).to(device)

    # encoder/decoder weights are randomly initilized
    # if checkpointed, load saved weights
    if args.load_checkpoint is not None:
        encoder.load_state_dict(state['enc_state'])
        decoder.load_state_dict(state['dec_state'])

    # read in datafiles
    train_pairs = split_lines(args.train_file)
    dev_pairs = split_lines(args.dev_file)
    test_pairs = split_lines(args.test_file)

    # set up optimization/loss
    params = list(encoder.parameters()) + list(decoder.parameters())  # .parameters() returns generator
    optimizer = optim.Adam(params, lr=args.initial_learning_rate)
    criterion = nn.NLLLoss()

    # optimizer may have state
    # if checkpointed, load saved state
    if args.load_checkpoint is not None:
        optimizer.load_state_dict(state['opt_state'])

    start = time.time()
    print_loss_total = 0  # Reset every args.print_every

    sorted_pairs = sorted(train_pairs, key=lambda pair: len(pair[0]))

    while iter_num < args.n_iters:
        iter_num += 1
        training_pair = tensors_from_pair(src_vocab, tgt_vocab, random.choice(train_pairs))
        input_tensor, input_lengths, target_tensor, target_lengths = getBatches(BATCH_SIZE,
                                                                                src_vocab, tgt_vocab, sorted_pairs)

        # input_tensor = training_pair[0]
        # target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, optimizer, criterion, input_lengths, target_lengths)
        print_loss_total += loss
        # print(loss)

        if iter_num % args.checkpoint_every == 0:
            state = {'iter_num': iter_num,
                     'enc_state': encoder.state_dict(),
                     'dec_state': decoder.state_dict(),
                     'opt_state': optimizer.state_dict(),
                     'src_vocab': src_vocab,
                     'tgt_vocab': tgt_vocab,
                     }
            filename = 'state_%010d.pt' % iter_num
            torch.save(state, filename)
            logging.debug('wrote checkpoint to %s', filename)

        if iter_num % args.print_every == 0:
            print_loss_avg = print_loss_total / args.print_every
            print_loss_total = 0
            logging.info('time since start:%s (iter:%d iter/n_iters:%d%%) loss_avg:%.4f',
                         time.time() - start,
                         iter_num,
                         iter_num / args.n_iters * 100,
                         print_loss_avg)
            # translate from the dev set
            translate_random_sentence(encoder, decoder, dev_pairs, src_vocab, tgt_vocab, n=2)
            translated_sentences = translate_sentences(encoder, decoder, dev_pairs, src_vocab, tgt_vocab)

            references = [[clean(pair[1]).split(), ] for pair in dev_pairs[:len(translated_sentences)]]
            candidates = [clean(sent).split() for sent in translated_sentences]
            dev_bleu = corpus_bleu(references, candidates)
            logging.info('Dev BLEU score: %.2f', dev_bleu)

    # translate test set and write to file
    translated_sentences = translate_sentences(encoder, decoder, test_pairs, src_vocab, tgt_vocab)
    with open(args.out_file, 'wt', encoding='utf-8') as outf:
        for sent in translated_sentences:
            outf.write(clean(sent) + '\n')

    # Visualizing Attention

    fig, ((ax, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    translate_and_show_attention("on p@@ eu@@ t me faire confiance .", encoder, decoder, src_vocab, tgt_vocab, fig, ax)
    translate_and_show_attention("j en suis contente .", encoder, decoder, src_vocab, tgt_vocab, fig, ax2)
    translate_and_show_attention("vous etes tres genti@@ ls .", encoder, decoder, src_vocab, tgt_vocab, fig, ax3)
    translate_and_show_attention("c est mon hero@@ s ", encoder, decoder, src_vocab, tgt_vocab, fig, ax4)
    plt.tight_layout()
    fig.savefig(args.fig)


def getBatches(batch_size, src_vocab, tgt_vocab, pairs):

    batches = []
    for i in range(batch_size):
        pair = random.choice(pairs)
        in_seq = [src_vocab.word2index[word] for word in pair[0].split(' ')]
        tgt_seq = [tgt_vocab.word2index[word] for word in pair[1].split(' ')]
        batches.append([in_seq, tgt_seq])

    in_seqs, tgt_seqs = zip(*sortPairs(batches))
    in_lengths = [len(s) for s in in_seqs]
    in_padded = [padding(s, max(in_lengths)) for s in in_seqs]
    tgt_lengths = [len(s) for s in tgt_seqs]
    tgt_padded = [padding(s, max(tgt_lengths)) for s in tgt_seqs]
    in_tensor = torch.tensor(in_padded, dtype=torch.long, device=device).transpose(0, 1)
    tgt_tensor = torch.tensor(tgt_padded, dtype=torch.long, device=device).transpose(0, 1)
    return in_tensor, in_lengths, tgt_tensor, tgt_lengths


def sortPairs(pairs):
    return sorted(pairs, key=lambda pair : len(pair[0]), reverse=True)


def padding(seq, max_length):
    seq += [PAD_index for i in range(max_length - len(seq))] + [EOS_index]
    return seq

if __name__ == '__main__':
    main()