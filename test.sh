CUDA_VISIBLE_DEVICES="0" \
python infer_sa_sam2.py \
--checkpoint "<set your checkpoint path here>" \
--test-images "<set your testing image dir here>" \
--test-masks "<set your testing mask dir here>" \
--save-dir "<set your prediction results dir here>"
