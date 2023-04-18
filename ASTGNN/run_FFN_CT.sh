#

outdir_base="../out/Ram/3channel/FFN_CT/exp2"
# scale_method_list=('time' 'time-channel')
# scale_method_list=('True' 'time' 'time-channel')
scale_method_list=('time')
# channel_target_list=("0" "1" "2")
channel_target_list=("0")

for scale_method in "${scale_method_list[@]}"
do

  STdatadir="../data/Devan/3channel/data_processed/all/scale-${scale_method}/samples.npz"

  for channel_target in "${channel_target_list[@]}"
  do

    outdir="${outdir_base}/scale-${scale_method}/channel_target-${channel_target}"
    python run_FFN_CT.py --STdatadir="${STdatadir}" \
                         --outdir="${outdir}" \
                         --channel_target="${channel_target}" \
                         --use_cuda=True

  done
done
