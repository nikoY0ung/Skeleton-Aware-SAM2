CUDA_VISIBLE_DEVICES="0" \
python train_sa_sam2.py \
--hiera-checkpoint "<set your pretrained hiera path here>" \
--train-images "<set your training image dir here>" \
--train-masks "<set your training mask dir here>" \
--save-dir "<set your checkpoint saving dir here>" \
--epoch 20 \
--lr 0.001 \
--batch_size 12
