from glob import glob
import argparse
from collections import defaultdict, Counter
from itertools import combinations, product, groupby
from pathlib import Path
import os
from sklearn.utils import shuffle
import numpy as np
import random
from shutil import copy
from subprocess import check_call

np.random.seed(42)
random.seed(42)


def get_fname(s):
    return s.split("\t")[0]

def get_emotion(s):
    return get_fname(s).split("_")[0].split("/")[1].lower()

def get_utt_id(s):
    return get_fname(s).split(".")[0].split("_")[-1]

def dedup(seq):
    """ >> remove_repetitions("1 2 2 3 100 2 2 1")
    '1 2 3 100 2 1' """
    seq = seq.strip().split(" ")
    result = seq[:1]
    reps = []
    rep_counter = 1
    for k in seq[1:]:
        if k != result[-1]:
            result += [k]
            reps += [rep_counter]
            rep_counter = 1
        else:
            rep_counter += 1
    reps += [rep_counter]
    assert len(reps) == len(result) and sum(reps) == len(seq)
    return " ".join(result) + "\n" #, reps

def remove_under_k(seq, k):
    """ remove tokens that repeat less then k times in a row
    >> remove_under_k("a a a a b c c c", 1) ==> a a a a c c c """
    seq = seq.strip().split(" ")
    result = []

    freqs = [(k,len(list(g))) for k, g in groupby(seq)]
    for c, f in freqs:
        if f > k:
            result += [c for _ in range(f)]
    return " ".join(result) + "\n" #, reps


def call(cmd):
    print(cmd)
    check_call(cmd, shell=True)


def denoising_preprocess(path, lang, dict):
    bin = 'fairseq-preprocess'
    cmd = [
        bin,
        f'--trainpref {path}/train.{lang} --validpref {path}/valid.{lang} --testpref {path}/test.{lang}',
        f'--destdir {path}/tokenized/{lang}',
        '--only-source',
        '--task multilingual_denoising',
        '--workers 40',
    ]
    if dict != "":
        cmd += [f'--srcdict {dict}']
    cmd = " ".join(cmd)
    call(cmd)


def translation_preprocess(path, src_lang, trg_lang, dict, only_train=False):
    bin = 'fairseq-preprocess'
    cmd = [
        bin,
        f'--source-lang {src_lang} --target-lang {trg_lang}',
        # f'--trainpref {path}/train',
        f'--destdir {path}/tokenized',
        '--workers 40',
        f'--testpref {path}/test'
    ]
    # if not only_train:
    #     cmd += [f'--validpref {path}/valid --testpref {path}/test']
    if dict != "":
        cmd += [
            f'--srcdict {dict}',
            f'--tgtdict {dict}',
        ]
    cmd = " ".join(cmd)
    call(cmd)


def load_tsv_km(tsv_path, km_path):
    print(f"loading {tsv_path} and {km_path}")
    assert tsv_path.exists() and km_path.exists()
    tsv_lines = open(tsv_path, "r").readlines()
    root, tsv_lines = tsv_lines[0], tsv_lines[1:]
    km_lines = open(km_path, "r").readlines()
    assert len(tsv_lines) == len(km_lines), ".tsv and .km should be the same length!"
    return root, tsv_lines, km_lines


def main(
    data,
    output_path,
    src_emotion,
    trg_emotion,
    dict_path,
    cross_speaker=True,
    autoencode=False,
    zero_shot=False,
    km_ext = "km",
):
    desc = """
    this script takes as input .tsv and .km files for EMOV dataset, and a pairs of emotions.
    it generates parallel .tsv and .km files for these emotions. for exmaple:
    ❯ python build_emov_translation_manifests.py \
            /checkpoint/felixkreuk/datasets/emov/manifests/emov_16khz/train.tsv \
            /checkpoint/felixkreuk/datasets/emov/manifests/emov_16khz/emov_16khz_km_100/train.km \
            ~/tmp/emov_pairs \
            --src-emotion amused --trg-emotion neutral \
            --dedup --shuffle --cross-speaker --dry-run
    """
    # SPEAKERS = ["bea", "jenie", "josh", "sam", "SAME"]
    # EMOTIONS = ['neutral', 'amused', 'angry', 'disgusted', 'sleepy']

    suffix = ""
    if cross_speaker: suffix += "_cross-speaker"
    if dedup: suffix += "_dedup"
    translation_suffix = ""
    if autoencode: translation_suffix += "_autoencode"
    denoising_suffix = ""
    denoising_suffix += "_zeroshot" if zero_shot else "_nonzeroshot"

    data = Path(data)
    translation_dir = Path(output_path)
    os.makedirs(translation_dir, exist_ok=True)

    root, tsv_lines, km_lines = load_tsv_km(
        tsv_path = data / "data.tsv",
        km_path = data / f"data.{km_ext}"
    )

    print("---")
    print(f"src emotions: {src_emotion}\ntrg emotions: {trg_emotion}")

    # create a dictionary with the following structure:
    # output[SPEAKER][UTT_ID] = list with indexes of line from the tsv file
    # that match the speaker and utterance id. for exmaple:
    # output = {'sam': {'0493': [875, 1608, 1822], ...}, ...}
    # meaning, for speaker 'sam', utterance id '0493', the indexes in tsv_lines
    # are 875, 1608, 1822
    spkr2utts = defaultdict(lambda: defaultdict(list))
    print(spkr2utts)
    for i, tsv_line in enumerate(tsv_lines):
        speaker = tsv_line.split("/")[0]
        if cross_speaker: speaker = "SAME"
        # assert speaker in SPEAKERS, "unknown speaker! make sure the .tsv contains EMOV data"
        utt_id = get_utt_id(tsv_line)
        spkr2utts[speaker][utt_id].append(i)
    print(spkr2utts)
        # create a tsv and km files with all the combinations for translation
    src_tsv, trg_tsv, src_km, trg_km = [], [], [], []
    for speaker, utt_ids in spkr2utts.items():
        for utt_id, indices in utt_ids.items():
            # generate all pairs
            pairs = [(x,y) for x in indices for y in indices]
            # print(len(pairs))
            # self-translation 
            if src_emotion == trg_emotion:
                pairs = [(x,y) for (x,y) in pairs if x == y]
            # filter according to src and trg emotions
            pairs = [(x,y) for (x,y) in pairs 
                    if get_emotion(tsv_lines[x]) == src_emotion.lower() and get_emotion(tsv_lines[y]) == trg_emotion.lower()]
            print(pairs)

            for idx1, idx2 in pairs:
                assert get_utt_id(tsv_lines[idx1]) == get_utt_id(tsv_lines[idx2])
                src_tsv.append(tsv_lines[idx1])
                trg_tsv.append(tsv_lines[idx2])
                km_line_idx1 = km_lines[idx1]
                km_line_idx2 = km_lines[idx2]
                km_line_idx1 = km_line_idx1 if not dedup else dedup(km_line_idx1)
                km_line_idx2 = km_line_idx2 if not dedup else dedup(km_line_idx2)
                src_km.append(km_line_idx1)
                trg_km.append(km_line_idx2)
    assert len(src_tsv) == len(trg_tsv) == len(src_km) == len(trg_km)
    print(f"{len(src_tsv)} pairs")

    if len(src_tsv) == 0:
        raise Exception("ERROR: generated 0 pairs!")

    # create files
    os.makedirs(translation_dir / f"{src_emotion}-{trg_emotion}", exist_ok=True)
    open(translation_dir / f"{src_emotion}-{trg_emotion}" / f"files.test.{src_emotion}", "w").writelines([root] + src_tsv)
    open(translation_dir / f"{src_emotion}-{trg_emotion}" / f"files.test.{trg_emotion}", "w").writelines([root] + trg_tsv)
    open(translation_dir / f"{src_emotion}-{trg_emotion}" / f"test.{src_emotion}", "w").writelines(src_km)
    open(translation_dir / f"{src_emotion}-{trg_emotion}" / f"test.{trg_emotion}", "w").writelines(trg_km)

    os.makedirs(translation_dir / "tokenized", exist_ok=True)

    translation_preprocess(translation_dir / f"{src_emotion}-{trg_emotion}", src_emotion, trg_emotion, dict_path)#, only_train=SRC_EMOTION==TRG_EMOTION)
    os.system(f"cp -rf {translation_dir}/**/tokenized/* {translation_dir}/tokenized")

if __name__ == "__main__":
    main(
        
    )
