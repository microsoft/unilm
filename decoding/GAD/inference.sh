data_dir=./data # the dir that contains dict files
checkpoint_path=./checkpoints/wmt14-en-de-base-nat-drafter-checkpoint.avg10.pt # the dir that contains NAT drafter checkpoint
AR_checkpoint_path=./checkpoints/wmt14-en-de-base-at-verifier.pt # the dir that contains AT verifier checkpoint
input_path=./test.en # the dir that contains bpe test files
output_path=./output/block.out # the dir for outputs

strategy='gad' # fairseq, AR, gad
batch=32
beam=5

beta=5
tau=3.0
block_size=25

src=en
tgt=de


python inference.py ${data_dir} --path ${checkpoint_path} \
      --user-dir block_plugins --task translation_lev_modified --remove-bpe --max-sentences 20 --source-lang ${src} \
      --target-lang ${tgt} --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 --iter-decode-with-beam 1 \
      --gen-subset test --AR-path ${AR_checkpoint_path} --input-path ${input_path} --output-path ${output_path} \
      --block-size ${block_size} --beta ${beta} --tau ${tau} --batch ${batch} --beam ${beam} --strategy ${strategy}
