$DATA = processed_data/

python3 dump_hubert_feature.py ./../../emotion_conversion/data data ./../../emotion_conversion/data/hubert_base_ls960.pt 9 1 0 ./../../emotion_conversion/data

python dump_km_label.py ./../../emotion_conversion/data data ./../../emotion_conversion/data/hubert_base_ls960_layer9_clusters200/data_hubert_base_ls960_layer9_clusters200.bin 1 0 ./../../emotion_conversion/data

python preprocess/create_core_manifest.py \
    --tsv data/data.tsv \
    --emov-km data/hubert_base_ls960_layer9_clusters200/data.km \
    --km data/hubert_base_ls960_layer9_clusters200/vctk.km \
    --dict data/hubert_base_ls960_layer9_clusters200/dict.txt \
    --manifests-dir processed_data

python preprocess/build_translation_manifests_perso.py \
    data/ \
    test/fairseq-data \
    --src-emotion neutral \
    --trg-emotion amused \
    -dd \
    -cs \
    --dict data/hubert_base_ls960_layer9_clusters200/dict.txt

python examples/emotion_conversion/preprocess/extract_f0.py \
    --tsv examples/emotion_conversion/data/data.tsv \
    --extractor pyaapt

python3 train.py --checkpoint_path ../../../../save/hifigan/ --config configs/EmoV/emov_hubert-layer9-cluster200_fixed-spkr-embedder_f0-raw_gst.json

python3 train.py \
    examples/emotion_conversion/processed_data/131123/fairseq-data/emov_multilingual_denoising_cross-speaker_dedup_nonzeroshot/tokenized \
    --distributed-world-size 1\
    --save-dir ../save/pretrain/ \
    --tensorboard-logdir ../logs/pretrain/ \
    --langs neutral,amused,angry,sleepy,disgusted,vctk.km \
    --dataset-impl mmap \
    --task multilingual_denoising \
    --arch transformer_small --criterion cross_entropy \
    --multilang-sampling-alpha 1.0 --sample-break-mode eos --max-tokens 16384 \
    --update-freq 1 --max-update 3000000 \
    --dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.0 \
    --optimizer adam --weight-decay 0.01 --adam-eps 1e-06 \
    --clip-norm 0.1 --lr-scheduler polynomial_decay --lr 0.0003 \
    --total-num-update 3000000 --warmup-updates 10000 --fp16 \
    --poisson-lambda 3.5 --mask 0.3 --mask-length span-poisson --replace-length 1 --rotate 0 --mask-random 0.1 --insert 0 --permute-sentences 1 \
    --skip-invalid-size-inputs-valid-test \
    --user-dir examples/emotion_conversion/fairseq_models

## Neutral to Amused

python train.py \
    --distributed-world-size 1 \
    examples/emotion_conversion/processed_data/131123/fairseq-data/emov_multilingual_translation_cross-speaker_dedup/tokenized/ \
    --save-dir ../save/train_neutral_amused \
    --tensorboard-logdir ../logs/train \
    --arch multilingual_small --task multilingual_translation \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
    --lang-pairs neutral-amused,amused-neutral \
    --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --lr 1e-05 --clip-norm 0 --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --warmup-updates 2000 --lr-scheduler inverse_sqrt \
    --max-tokens 4096 --update-freq 1 --max-update 100000 \
    --required-batch-size-multiple 8 --fp16 --num-workers 4 \
    --seed 2 --log-format json --log-interval 25 --save-interval-updates 1000 \
    --no-epoch-checkpoints --keep-best-checkpoints 1 --keep-interval-updates 1 \
    --finetune-from-model ../save/pretrain/checkpoint_best.pt \
    --user-dir examples/emotion_conversion/fairseq_models

## Neutral to Angry

python train.py \
    --distributed-world-size 1 \
    examples/emotion_conversion/processed_data/211123/fairseq-data/emov_multilingual_translation_cross-speaker_dedup/tokenized/ \
    --save-dir ../save/train_neutral_angry \
    --tensorboard-logdir ../logs/train \
    --arch multilingual_small --task multilingual_translation \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
    --lang-pairs neutral-angry,angry-neutral \
    --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --lr 1e-05 --clip-norm 0 --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --warmup-updates 2000 --lr-scheduler inverse_sqrt \
    --max-tokens 4096 --update-freq 1 --max-update 100000 \
    --required-batch-size-multiple 8 --fp16 --num-workers 4 \
    --seed 2 --log-format json --log-interval 25 --save-interval-updates 1000 \
    --no-epoch-checkpoints --keep-best-checkpoints 1 --keep-interval-updates 1 \
    --finetune-from-model ../save/pretrain/checkpoint_best.pt \
    --user-dir examples/emotion_conversion/fairseq_models

## Neutral to sleepy

python train.py \
    --distributed-world-size 1 \
    examples/emotion_conversion/processed_data/211123/fairseq-data/emov_multilingual_translation_cross-speaker_dedup/tokenized/ \
    --save-dir ../save/train_neutral_sleepy \
    --tensorboard-logdir ../logs/train \
    --arch multilingual_small --task multilingual_translation \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
    --lang-pairs neutral-sleepy,sleepy-neutral \
    --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --lr 1e-05 --clip-norm 0 --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --warmup-updates 2000 --lr-scheduler inverse_sqrt \
    --max-tokens 4096 --update-freq 1 --max-update 100000 \
    --required-batch-size-multiple 8 --fp16 --num-workers 4 \
    --seed 2 --log-format json --log-interval 25 --save-interval-updates 1000 \
    --no-epoch-checkpoints --keep-best-checkpoints 1 --keep-interval-updates 1 \
    --finetune-from-model ../save/pretrain/checkpoint_best.pt \
    --user-dir examples/emotion_conversion/fairseq_models

## Neutral to disgusted

python train.py \
    --distributed-world-size 1 \
    examples/emotion_conversion/processed_data/211123/fairseq-data/emov_multilingual_translation_cross-speaker_dedup/tokenized/ \
    --save-dir ../save/train_neutral_disgusted \
    --tensorboard-logdir ../logs/train \
    --arch multilingual_small --task multilingual_translation \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
    --lang-pairs neutral-disgusted,disgusted-neutral \
    --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --lr 1e-05 --clip-norm 0 --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --warmup-updates 2000 --lr-scheduler inverse_sqrt \
    --max-tokens 4096 --update-freq 1 --max-update 100000 \
    --required-batch-size-multiple 8 --fp16 --num-workers 4 \
    --seed 2 --log-format json --log-interval 25 --save-interval-updates 1000 \
    --no-epoch-checkpoints --keep-best-checkpoints 1 --keep-interval-updates 1 \
    --finetune-from-model ../save/pretrain/checkpoint_best.pt \
    --user-dir examples/emotion_conversion/fairseq_models

python -m emotion_models.pitch_predictor n_tokens=200 \
    train_tsv="/home/jovyan/emotion/fairseq/examples/emotion_conversion/processed_data/131123/denoising/emov/train.tsv" \
    train_km="/home/jovyan/emotion/fairseq/examples/emotion_conversion/processed_data/131123/denoising/emov/train.km" \
    valid_tsv="/home/jovyan/emotion/fairseq/examples/emotion_conversion/processed_data/131123/denoising/emov/valid.tsv" \
    valid_km="/home/jovyan/emotion/fairseq/examples/emotion_conversion/processed_data/131123/denoising/emov/valid.km"

for emotion in "neutral" "amused" "angry" "disgusted" "sleepy"; do
    python -m emotion_models.duration_predictor n_tokens=200 substring=$emotion \
        train_tsv="/home/jovyan/emotion/fairseq/examples/emotion_conversion/processed_data/131123/denoising/emov/train.tsv" \
        train_km="/home/jovyan/emotion/fairseq/examples/emotion_conversion/processed_data/131123/denoising/emov/train.km" \
        valid_tsv="/home/jovyan/emotion/fairseq/examples/emotion_conversion/processed_data/131123/denoising/emov/valid.tsv" \
        valid_km="/home/jovyan/emotion/fairseq/examples/emotion_conversion/processed_data/131123/denoising/emov/valid.km"
done


GENERATION

fairseq-generate \
    examples/emotion_conversion/processed_data/131123/fairseq-data/emov_multilingual_translation_cross-speaker_dedup/tokenized/ \
    --task multilingual_translation \
    --gen-subset test \
    --path /home/jovyan/emotion/save/train_neutral_amused/checkpoint_best.pt \
    --beam 5 \
    --batch-size 4 --max-len-a 1.8 --max-len-b 10 --lenpen 1 --min-len 1 \
    --skip-invalid-size-inputs-valid-test --distributed-world-size 1 \
    --source-lang neutral --target-lang amused \
    --lang-pairs neutral-amused,amused-neutral \
    --results-path examples/emotion_conversion/results/token/ \
    --user-dir /home/jovyan/emotion/fairseq/examples/emotion_conversion/fairseq_models


fairseq-generate \
    examples/emotion_conversion/test/fairseq-data/emov_multilingual_translation_cross-speaker_dedup/tokenized/ \
    --task multilingual_translation \
    --gen-subset test \
    --path examples/emotion_conversion/save/train_neutral_amused/checkpoint_best.pt \
    --beam 5 \
    --batch-size 4 --max-len-a 1.8 --max-len-b 10 --lenpen 1 --min-len 1 \
    --skip-invalid-size-inputs-valid-test --distributed-world-size 1 \
    --source-lang neutral --target-lang amused \
    --lang-pairs neutral-amused,amused-neutral \
    --results-path examples/emotion_conversion/results/token/ \
    --user-dir examples/emotion_conversion/fairseq_models

python examples/emotion_conversion/synthesize.py \
    --result-path examples/emotion_conversion/results/token/generate-test.txt \
    --data examples/emotion_conversion/test/fairseq-data/emov_multilingual_translation_cross-speaker_dedup/neutral-amused \
    --orig-tsv examples/emotion_conversion/data/data.tsv \
    --orig-km examples/emotion_conversion/data/data.km \
    --checkpoint-file examples/emotion_conversion/save/hifigan/g_00170000 \
    --dur-model examples/emotion_conversion/save/duration_predictor/ \
    --f0-model examples/emotion_conversion/save/pitch_predictor/pitch_predictor.ckpt \
    -s neutral -t amused \
    --outdir examples/emotion_conversion/results/wavs/neutral-amused


python3 examples/emotion_conversion/preprocessing.py \
    --model-dir /home/utilisateur/createch/project/emotion/fairseq/examples/emotion_conversion/save \
    --data /home/utilisateur/createch/project/emotion/fairseq/examples/emotion_conversion/data \
    --split data \
    --output-path /home/utilisateur/createch/project/emotion/fairseq/examples/emotion_conversion/processed_data_test \
    --src-emotion neutral \
    --trg-emotion amused \
    --dict /home/utilisateur/createch/project/emotion/fairseq/examples/emotion_conversion/data/dict.txt \
    --user-dir /home/utilisateur/createch/project/emotion/fairseq/examples/emotion_conversion/fairseq_models \
    --dataset /home/utilisateur/createch/project/emotion/dataset_final