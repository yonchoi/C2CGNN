# ==============================================================================
#  Ram replicate data
# ==============================================================================

### Set up for single channel data
outdir_base="out/ram/replicate/exp4/all"

treatment_list=("all")
input_list=("ERK" "param")
# input_list=("ERK")
# target_list=('EGR1' 'PERK')
target_list=('EGR1' 'PERK' 'cJun' 'cMyc' 'pcFos' 'cFos' 'pRB' 'FRA1')
# target_list=('EGR1' 'PERK')
anndatadir="data/Ram/All_Replicates_dataset/combined_data_raw.h5ad"

for target in "${target_list[@]}"
  do

  for input in "${input_list[@]}"
  do

    for treatment in "${treatment_list[@]}"
    do

      if [ $input == "ERK" ]
      then
        subset_list=("all" "+70" "-70")
      else
        subset_list=("all")
      fi

        for input_subset in "${subset_list[@]}"
        do

          outdir="${outdir_base}/${treatment}/${input}_${input_subset}/${target}"
          python -m experiments.ETG_prediction.run_evaluation.py --anndatadir="${anndatadir}" \
                                                                 --treatment="${treatment}" \
                                                                 --outdir="${outdir}" \
                                                                 --input="${input}" \
                                                                 --output="Y" \
                                                                 --target="${target}" \
                                                                 --use_cuda="True" \
                                                                 --do_attribution="True" \
                                                                 --input_subset="${input_subset}"
        done

    done

  done

done

# Aggregate results
python -m experiments.ETG_prediction.combine_evaluation.py --outdir="${outdir_base}" \
                                                           --do_attribution="True"

# ==============================================================================
#  Ram replicate data
# ==============================================================================

### Set up for single channel data
outdir_base="out/ram/replicate/exp5/all/param"
# outdir_base="out/ram/replicate/exp5/all/ERK"

treatment_list=("all")
input_list=("Y")
# input_list=("ERK")
# target_list=('EGR1' 'PERK')
target_list=('pca0' 'pca1' 'pca2')
target_list=('Mean' 'Max' 'Mean_deriv' 'Mean_pk' 'Mean_amp' 'Dur' 'Sum_dur' 'Period' 'Freq')
# target_list=('EGR1' 'PERK')
anndatadir="data/Ram/All_Replicates_dataset/combined_data_raw.h5ad"
subset_list=('EGR1' 'PERK' 'cJun' 'cMyc' 'pcFos' 'cFos' 'pRB' 'FRA1')

for target in "${target_list[@]}"
 do

 for input in "${input_list[@]}"
 do

   for treatment in "${treatment_list[@]}"
   do

     for input_subset in "${subset_list[@]}"
     do

       outdir="${outdir_base}/${treatment}/${input}_${input_subset}/${target}"
       python -m experiments.ETG_prediction.run_evaluation.py --anndatadir="${anndatadir}" \
                                                              --treatment="${treatment}" \
                                                              --outdir="${outdir}" \
                                                              --input="${input}" \
                                                              --output="param" \
                                                              --target="${target}" \
                                                              --use_cuda="True" \
                                                              --do_attribution="True" \
                                                              --input_subset="${input_subset}"
     done

   done

 done

done


# Aggregate results
python -m experiments.ETG_prediction.combine_evaluation.py --outdir="${outdir_base}" \
                                                           --do_attribution="True"

# ==============================================================================
#  Ram replicate data subset
# ==============================================================================

### Set up for single channel data
outdir="out/ram/replicate_subset/exp"

treatment_list=("all")

input_list=("ERK" "ERK_P")
target_list=('EGR1' 'PERK' 'cJun' 'cMyc' 'pcFos' 'cFos' 'pRB' 'FRA1')
# target_list=('EGR1' 'PERK')

anndatadir="data/Ram/All_Replicates_dataset/combined_data_raw_subset.h5ad"

for target in "${target_list[@]}"
  do

  for input in "${input_list[@]}"
  do

    for treatment in "${treatment_list[@]}"
    do
      python -m experiments.ETG_prediction.run_evaluation.py --anndatadir="${anndatadir}" \
                                                             --treatment="${treatment}" \
                                                             --outdir="${outdir}" \
                                                             --input="${input}" \
                                                             --target="${target}" \
                                                             --use_cuda="True" \
                                                             --do_attribution="True"
    done

  done

done

# Aggregate results
python -m experiments.ETG_prediction.combine_evaluation.py --outdir="${outdir}" \
                                                           --do_attribution="True"


# ==============================================================================
#  Ram initial data
# ==============================================================================

### Set up for single channel data
# outdir="out/ram/exp10_scaled-128"
outdir="out/ram/exp13_scaled-128"

# treatment_list=("all")
# treatment_list=("EGF31.64ng.ml EGF0.01ng.ml")
treatment_list=("EGF0.01ng.ml")
# input_list=("ERK" "ERK_P" "ERK-ERK_P")
input_list=("ERK")
# target_list=('EGR1' 'pERK' 'FRA1' 'pBR1')
# target_list=('EGR1' 'pERK' 'FRA1' 'pBR')
# target_list=('EGR1' 'pERK' 'FRA1' 'pRB')
target_list=('pRB')
anndatadir="data/Ram/ERK_ETGs_Replicate1/combined_data_raw.h5ad"

for target in "${target_list[@]}"
  do

  for input in "${input_list[@]}"
  do

    for treatment in "${treatment_list[@]}"
    do
      python -m experiments.ETG_prediction.run_evaluation.py --anndatadir="${anndatadir}" \
                                                             --treatment="${treatment}" \
                                                             --outdir="${outdir}" \
                                                             --input="${input}" \
                                                             --target="${target}" \
                                                             --use_cuda="True"
    done

  done

done

# Aggregate results
python -m experiments.ETG_prediction.combine_evaluation.py --outdir="${outdir}"
python -m experiments.ETG_prediction.run_attribution.py
python -m experiments.ETG_prediction.individual_attr.py

# ==============================================================================
#  Set up for 3-channel data
# ==============================================================================
outdir_base="out/3channel/exp10"
anndatadir="data/Devan/3channel/combined_data_raw.h5ad"

# treatment_list=("all")
treatment_list=("EGF 100ng.mL16tpvehicle 61tp" "MK 500nM16tpIL-1b 1ng.mL61tp" 'EGF 1ng.mL16tpvehicle 61tp' 'veh 16tpvehicle 61tp')
# input_list=("ERK_P")
input_list=("ERK" "ERK_P")
target_list=('IL8')
transform_y_list=("scale" "log" "invnorm")
# remove_outlier_list=("0" "0.01" "0.03" "0.05")
remove_outlier_list=("0.05")

for remove_outlier in "${remove_outlier_list[@]}"
  do

  for transform_y in "${transform_y_list[@]}"
    do

    outdir="${outdir_base}/${transform_y}/${remove_outlier}"

    for target in "${target_list[@]}"
      do

      for input in "${input_list[@]}"
      do

        for treatment in "${treatment_list[@]}"
        do
          python -m experiments.ETG_prediction.run_evaluation.py --anndatadir="${anndatadir}" \
                                                                 --treatment="${treatment}" \
                                                                 --outdir="${outdir}" \
                                                                 --input="${input}" \
                                                                 --target="${target}" \
                                                                 --use_cuda="True" \
                                                                 --transform_y="${transform_y}" \
                                                                 --remove_outlier="${remove_outlier}"
        done

      done

    done

    python -m experiments.ETG_prediction.combine_evaluation.py --outdir="${outdir}"

  done

done

# Aggregate results
python -m experiments.ETG_prediction.combine_evaluation.py --outdir="${outdir}"
# python -m experiments.ETG_prediction.run_attribution.py
# python -m experiments.ETG_prediction.individual_attr.py
