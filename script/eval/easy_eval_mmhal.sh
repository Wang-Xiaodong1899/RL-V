ckpt_path=$1
save_dir=$2

q_file=./eval/data/mmhal-bench_with_image.jsonl
template_file=./eval/data/mmhal-bench_answer_template.json
answer_file_name=mmhal-bench_answer.jsonl

answer_file=$save_dir/$answer_file_name

python ./muffin/eval/muffin_vqa.py \
            --model-path $ckpt_path \
            --question-file $q_file \
            --answers-file $answer_file \
            # --temperature 0

echo "generate vqa answer done!"

python ./eval/change_mmhal_predict_template.py \
    --response-template ./eval/data/mmhal-bench_answer_template.json \
    --answers-file $answer_file \
    --save-file ${answer_file}.template.json

echo "change response by template done"

python ./eval/eval_gpt_mmhal.py \
    --response $answer_file.template.json \
    --evaluation $answer_file.mmhal_test_eval.json \
    >> ${answer_file}.eval_log.txt

echo "GPT eval done , please check the eval_log.txt"