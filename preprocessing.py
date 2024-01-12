import argparse
from pathlib import Path
from token_preprocessing.dump_hubert_feature import main as dump_feature
from token_preprocessing.dump_km_label_personize import dump_label
from preprocess.build_translation_manifests_perso import main as create_manifest
from preprocess.extract_f0 import main as extract_f0
from subprocess import check_call
from synthesize_perso import synth as synthesize
import os
import shutil

def call(
    preprocessed_data_dir,
    model_checkpoint,
    src_emotion,
    trg_emotion,
    result_path,
    user_dir,
):
    combination = f"{src_emotion}-{trg_emotion},{trg_emotion}-{src_emotion}"
    preprocessed_data_dir = preprocessed_data_dir  / "tokenized"
    cmd = f"""fairseq-generate \
    {preprocessed_data_dir} \
    --task multilingual_translation \
    --gen-subset test \
    --path {model_checkpoint} \
    --beam 5 \
    --batch-size 4 --max-len-a 1.8 --max-len-b 10 --lenpen 1 --min-len 1 \
    --skip-invalid-size-inputs-valid-test --distributed-world-size 1 \
    --source-lang {src_emotion} --target-lang {trg_emotion} \
    --lang-pairs {combination} \
    --results-path {result_path} \
    --user-dir {user_dir}
    """
    check_call(cmd, shell=True)



def main(args):
    #Preprocessing
    dump_feature(
        args.data, 
        args.split, 
        os.path.join(args.model_dir, "hubert", "hubert_base_ls960.pt"),
        args.layer, 
        args.nshard, 
        args.rank, 
        args.data, 
        args.max_chunk
        )
    dump_label(
        args.data, 
        args.split, 
        os.path.join(args.model_dir, "km_model", "data_hubert_base_ls960_layer9_clusters200.bin"),
        args.nshard, 
        args.rank, 
        args.data
        )
    #work
        
    #Creation of the manifest
    create_manifest(
        args.data, 
        args.output_path, 
        args.src_emotion, 
        args.trg_emotion,
        args.dict,
        args.cross_speaker, 
        args.autoencode, 
        args.zero_shot, 
        args.km_ext, 
        )
    #work
    
    #Extract F0
    # extract_f0(
    #     os.path.join(args.data, f"{args.split}.tsv"), 
    #     args.n_workers
    # )
    #work
    print("----------------create mafnifest finish----------------")
    
    #generate token
    call(
        args.output_path, 
        os.path.join(args.model_dir, f"train_{args.src_emotion}_{args.trg_emotion}", "checkpoint_best.pt"),
        args.src_emotion, 
        args.trg_emotion, 
        os.path.join(args.output_path, "token"), 
        args.user_dir
        )
    # work
    
    #synthesize
    synthesize(
        os.path.join(args.model_dir, "hifigan", "g_00400000"),
        "cuda:0",
        os.path.join(args.model_dir, "duration_predictor"),
        os.path.join(args.model_dir, "pitch_predictor", "pitch_predictor.ckpt"),
        os.path.join(args.output_path, "token", "generate-test.txt"), 
        args.output_path / f"{args.src_emotion}-{args.trg_emotion}",
        args.src_emotion,
        args.trg_emotion,
        os.path.join(args.data, f"{args.split}.tsv"),
        os.path.join(args.data, f"{args.split}.km"),
        args.dataset
    )
    
    shutil.rmtree(os.path.join(args.output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, help="path to model directory")
    #preprocessing
    parser.add_argument("--data",required=False, default="./../../emotion_conversion/data", help="path to data directory")
    parser.add_argument("--split", default="data", help="data split to preprocess")
    parser.add_argument("--layer", type=int, default=9)
    parser.add_argument("--nshard", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--feat_dir", default="data")
    parser.add_argument("--max_chunk", type=int, default=1600000)
    
    #Creation of the manifest
    parser.add_argument("--output-path", type=Path, help="output directory with the manifests will be created")
    parser.add_argument("--src-emotion", type= str, help="emotion to translate from")
    parser.add_argument("--trg-emotion",type= str, help="emotion to translate to")
    parser.add_argument("-cs", "--cross-speaker", action='store_true', default=True, help="if set then translation will occur also between speakers, meaning the same sentence can be translated between different speakers (default: false)")
    parser.add_argument("-dd", "--dedup", action='store_true', default=True, help="remove repeated tokens (example: 'aaabc=>abc')")
    parser.add_argument("-sh", "--shuffle", action='store_true', default=False, help="shuffle the data")
    parser.add_argument("-ae", "--autoencode", action='store_true', default=False, help="include training pairs from the same emotion (this includes examples of the same sentence uttered by different people and examples where the src and trg are the exact same seq)")
    parser.add_argument("-dr", "--dry-run", action='store_true', default=False, help="don't write anything to disk")
    parser.add_argument("-zs", "--zero-shot", action='store_true', default=False, help="if true, the denoising task will train on the same splits as the translation task (split by utterance id). if false, the denoising task will train on randomly sampled splits (not split by utterance id)")
    parser.add_argument("--km-ext", default="km", help="")
    parser.add_argument("--dict", default="/checkpoint/felixkreuk/datasets/emov/manifests/emov_16khz/fairseq.dict.txt", help="")
    
    #Extract F0
    parser.add_argument("--n_workers", type=int, default=40, help="")
    
    
    #Generate token
    
    parser.add_argument('--user-dir', type=Path, help='path to fairseq user directory')
    
    #Synthesize
        
    parser.add_argument('--device', type=int, default=0)
    
    parser.add_argument('--dataset', type=str, default='./../dataset', help='path to dataset')
    args = parser.parse_args()

    main(args)