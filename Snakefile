# Stages 3 (analysis) and 4 (figures/tables). Stages 1 and 2 have their own
# Snakefiles under 1_data-preprocessing/ and 2_model-finetuning/.
#
#   snakemake --cores 8 figures     # stage 4 only, from shipped fig_table_gen/
#   snakemake --cores 8 analysis    # stage 3, rebuilding fig_table_gen/
#   snakemake --cores 8 all

import os

WORKDIR = os.path.dirname(os.path.abspath(workflow.snakefile))
workdir: WORKDIR

DATA = "data"
GEN  = f"{DATA}/fig_table_gen"
OUT  = "output"


# --------------------------- stage 3: analysis ---------------------------

rule collect_val_var_year_pairs:
    shell:
        "cd 3_analysis-prediction && python collect_val_var_year_pairs.py"

rule prep_accuracythresholds:
    output:
        "data/fig_table_gen/accuracythresholds_alpaca.csv",
    shell:
        "cd 3_analysis-prediction && python prep_accuracythresholds.py"

rule prep_alpaca_examples:
    input:
        "data/gss_train_vars_corrected_binarized_again.parquet",
    output:
        "data/fig_table_gen/alpaca_examples.parquet",
    shell:
        "cd 3_analysis-prediction && python prep_alpaca_examples.py"

rule prep_alpaca_vs_mf_groupauc:
    input:
        "data/predictions/maicomputer_alpaca-native_impute_10_128_50__resample1_wide.parquet",
        "data/predictions/maicomputer_alpaca-native_partial_10_128_50__resample1_wide.parquet",
        "data/predictions/mf_impute_10_128_50__resample1_wide.parquet",
        "data/predictions/mf_partial_10_128_50__resample1_wide.parquet",
        "data/df_analysis_slim.pkl",
        "data/gss_df_merged.parquet",
    output:
        "data/fig_table_gen/alpaca_vs_mf_groupauc.csv",
    shell:
        "cd 3_analysis-prediction && python prep_alpaca_vs_mf_groupauc.py"

rule prep_alpaca_vs_mf_individualauc:
    input:
        "data/predictions/maicomputer_alpaca-native_impute_10_128_50__resample1_wide.parquet",
        "data/predictions/maicomputer_alpaca-native_partial_10_128_50__resample1_wide.parquet",
        "data/predictions/mf_impute_10_128_50__resample1_wide.parquet",
        "data/predictions/mf_partial_10_128_50__resample1_wide.parquet",
        "data/df_analysis_slim.pkl",
        "data/gss_df_merged.parquet",
    output:
        "data/fig_table_gen/alpaca_vs_mf_individualauc_per_respondent.parquet",
    shell:
        "cd 3_analysis-prediction && python prep_alpaca_vs_mf_individualauc.py"

rule prep_auc_by_agreement:
    input:
        "data/predictions/maicomputer_alpaca-native_impute_10_128_50__resample1_wide.parquet",
        "data/df_analysis_slim.pkl",
    output:
        "data/fig_table_gen/auc_by_agreement.parquet",
    shell:
        "cd 3_analysis-prediction && python prep_auc_by_agreement.py"

rule prep_auc_by_nq:
    input:
        "data/df_analysis_slim.pkl",
        "data/predictions/maicomputer_alpaca-native_impute_0_10_128_50__resample1_long.parquet",
    output:
        "data/fig_table_gen/auc_by_nq.parquet",
    shell:
        "cd 3_analysis-prediction && python prep_auc_by_nq.py"

rule prep_concat_mf:
    output:
        "data/fig_table_gen/concat_mf.parquet",
    shell:
        "cd 3_analysis-prediction && python prep_concat_mf.py"

rule prep_counterfactual:
    input:
        "data/predictions/maicomputer_alpaca-native_partial_10_128_50__resample1_varyear.parquet",
        "data/predictions/mf_partial_10_128_50__resample1_varyear.parquet",
        "data/predictions/maicomputer_alpaca-native_partial_10_128_50__resample1_wide.parquet",
        "data/predictions/mf_partial_10_128_50__resample1_wide.parquet",
        "data/gss_df_merged.parquet",
        "data/df_analysis_slim.pkl",
        "data/gss_train_vars_corrected_binarized_again.parquet",
    output:
        "data/fig_table_gen/counterfactual_pop_mean.parquet",
        "data/fig_table_gen/counterfactual_variable_meta.parquet",
    shell:
        "cd 3_analysis-prediction && python prep_counterfactual.py"

rule prep_demographic_cv:
    input:
        "data/df_analysis_slim.pkl",
    output:
        "data/fig_table_gen/demographic_cv.parquet",
    shell:
        "cd 3_analysis-prediction && python prep_demographic_cv.py"

rule prep_distance_nearest_year:
    input:
        "data/fig_table_gen/retro_interp_forecast_varyear.parquet",
    output:
        "data/fig_table_gen/distance_nearest_year.parquet",
    shell:
        "cd 3_analysis-prediction && python prep_distance_nearest_year.py"

rule prep_embeddings:
    input:
        "data/df_analysis_slim.pkl",
        "data/weights/maicomputer_alpaca-native.pkl",
        "data/models/maicomputer_alpaca-native_partial_k0_10_128_50__resample1_best.weights.h5",
    output:
        "data/fig_table_gen/embeddings_question.parquet",
        "data/fig_table_gen/embeddings_respondent.parquet",
        "data/fig_table_gen/embeddings_period.parquet",
    shell:
        "cd 3_analysis-prediction && python prep_embeddings.py"

rule prep_featureimportance:
    input:
        "data/df_analysis.pkl",
        "data/weights/maicomputer_alpaca-native.pkl",
        "data/models/maicomputer_alpaca-native_partial_k2_10_128_50__resample1_best.weights.h5",
    output:
        "data/fig_table_gen/featureimportance_permutation.parquet",
    shell:
        "cd 3_analysis-prediction && python prep_featureimportance.py"

rule prep_fig_groupvar:
    input:
        "data/predictions/maicomputer_alpaca-native_partial_10_128_50__resample1_wide.parquet",
        "data/df_analysis_slim.pkl",
        "data/df_analysis.pkl",
        "data/gss_df_merged.parquet",
    output:
        "data/fig_table_gen/fig_groupvar_between.parquet",
        "data/fig_table_gen/fig_groupvar_between_summary.csv",
        "data/fig_table_gen/fig_groupvar_within.parquet",
        "data/fig_table_gen/fig_groupvar_within_summary.csv",
    shell:
        "cd 3_analysis-prediction && python prep_fig_groupvar.py"

rule prep_frontier_llm:
    output:
        "data/fig_table_gen/frontier_llm_metrics.csv",
    shell:
        "cd 3_analysis-prediction && python prep_frontier_llm.py"

rule prep_individualauc:
    input:
        "data/predictions/maicomputer_alpaca-native_impute_10_128_50__resample1_wide.parquet",
        "data/predictions/maicomputer_alpaca-native_partial_10_128_50__resample1_wide.parquet",
        "data/predictions/maicomputer_alpaca-native_total_10_128_50__resample1_wide.parquet",
        "data/df_analysis_slim.pkl",
        "data/gss_df_merged.parquet",
    output:
        "data/fig_table_gen/individualauc_per_respondent.parquet",
    shell:
        "cd 3_analysis-prediction && python prep_individualauc.py"

rule prep_missing_mech:
    output:
        "data/fig_table_gen/missing_mech_auc.parquet",
    shell:
        "cd 3_analysis-prediction && python prep_missing_mech.py"

rule prep_missing_prop:
    output:
        "data/fig_table_gen/missing_prop_auc.parquet",
    shell:
        "cd 3_analysis-prediction && python prep_missing_prop.py"

rule prep_modelcomparison:
    output:
        "data/fig_table_gen/modelcomparison_metrics.csv",
    shell:
        "cd 3_analysis-prediction && python prep_modelcomparison.py"

rule prep_modelperformance:
    input:
        "data/processed/figure4_roc_data.parquet",
        "data/predictions/maicomputer_alpaca-native_impute_10_128_50__resample1_varyear.parquet",
        "data/predictions/maicomputer_alpaca-native_partial_10_128_50__resample1_varyear.parquet",
        "data/predictions/maicomputer_alpaca-native_total_10_128_50__resample1_varyear.parquet",
    output:
        "data/fig_table_gen/modelperformance_roc_downsampled.parquet",
        "data/fig_table_gen/modelperformance_roc_auc.csv",
        "data/fig_table_gen/modelperformance_scatter.parquet",
    shell:
        "cd 3_analysis-prediction && python prep_modelperformance.py"

rule prep_opinionauc:
    input:
        "data/predictions/maicomputer_alpaca-native_impute_10_128_50__resample1_wide.parquet",
        "data/predictions/maicomputer_alpaca-native_partial_10_128_50__resample1_wide.parquet",
        "data/predictions/maicomputer_alpaca-native_total_10_128_50__resample1_wide.parquet",
        "data/df_analysis_slim.pkl",
        "data/df_analysis.pkl",
        "data/gss7221_r2.dta",
        "data/gss_train_vars_corrected_binarized_again.parquet",
        "data/weights/maicomputer_alpaca-native.pkl",
    output:
        "data/fig_table_gen/opinionauc_per_varyear.parquet",
    shell:
        "cd 3_analysis-prediction && python prep_opinionauc.py"

rule prep_perfbygroup:
    input:
        "data/predictions/maicomputer_alpaca-native_partial_10_128_50__resample1_varyear.parquet",
        "data/predictions/mf_partial_10_128_50__resample1_varyear.parquet",
    output:
        "data/fig_table_gen/perfbygroup_metrics.csv",
    shell:
        "cd 3_analysis-prediction && python prep_perfbygroup.py"

rule prep_period_controls:
    input:
        "data/df_analysis_slim.pkl",
    output:
        "data/fig_table_gen/_period_controls.parquet",
    shell:
        "cd 3_analysis-prediction && python prep_period_controls.py"

rule prep_polviews_rank:
    output:
        "data/fig_table_gen/polviews_rank.parquet",
    shell:
        "cd 3_analysis-prediction && python prep_polviews_rank.py"

rule prep_response_category_auc:
    input:
        "data/predictions/maicomputer_alpaca-native_impute_10_128_50__resample1_wide.parquet",
        "data/df_analysis_slim.pkl",
        "data/gss_train_vars_corrected_binarized_again.parquet",
    output:
        "data/fig_table_gen/response_category_auc.parquet",
    shell:
        "cd 3_analysis-prediction && python prep_response_category_auc.py"

rule prep_retro_distance_metrics:
    output:
        "data/fig_table_gen/retro_distance_metrics.csv",
    shell:
        "cd 3_analysis-prediction && python prep_retro_distance_metrics.py"

rule prep_retro_interp_forecast:
    output:
        "data/fig_table_gen/retro_interp_forecast_varyear.parquet",
    shell:
        "cd 3_analysis-prediction && python prep_retro_interp_forecast.py"

rule prep_retro_interp_forecast_metrics:
    output:
        "data/fig_table_gen/retro_interp_forecast_metrics.csv",
    shell:
        "cd 3_analysis-prediction && python prep_retro_interp_forecast_metrics.py"

rule prep_samesex_exclude:
    input:
        "data/df_analysis.pkl",
        "data/weights/maicomputer_alpaca-native.pkl",
        "data/fig_table_gen/maicomputer_alpaca-native_partial_k3_10_128_50__resample1[_excludeN]_best.weights.h5",
    output:
        "data/fig_table_gen/samesex_exclude.parquet",
    shell:
        "cd 3_analysis-prediction && python prep_samesex_exclude.py"

rule prep_samesexframing:
    input:
        "data/predictions/maicomputer_alpaca-native_partial_10_128_50__resample1_varyear.parquet",
        "data/predictions/maicomputer_alpaca-native_partial_10_128_50__resample1_wide.parquet",
    output:
        "data/fig_table_gen/samesexframing_paired_individual.parquet",
    shell:
        "cd 3_analysis-prediction && python prep_samesexframing.py"

rule prep_similar_questions:
    input:
        "data/weights/maicomputer_alpaca-native.pkl",
        "data/fig_table_gen/maicomputer_alpaca-native_partial_kK_10_128_50__resample1_best.weights.h5",
        "data/df_analysis_slim.pkl",
        "data/gss_train_vars_corrected_binarized_again.parquet",
    output:
        "data/fig_table_gen/similar_questions.parquet",
    shell:
        "cd 3_analysis-prediction && python prep_similar_questions.py"

rule prep_sorting_by_year:
    input:
        "data/fig_table_gen/opinionauc_per_varyear.parquet",
    output:
        "data/fig_table_gen/sorting_by_year.parquet",
    shell:
        "cd 3_analysis-prediction && python prep_sorting_by_year.py"

rule prep_top20_modules:
    input:
        "data/gss_train_vars_corrected_binarized_again.parquet",
        "data/module_dict.pkl",
        "data/df_analysis_slim.pkl",
    output:
        "data/fig_table_gen/top20_modules.parquet",
    shell:
        "cd 3_analysis-prediction && python prep_top20_modules.py"

rule prep_top50_samesex_corr:
    input:
        "data/corr_with_marhomo1.parquet",
        "data/gss_train_vars_corrected_binarized_again.parquet",
    output:
        "data/fig_table_gen/top50_samesex_corr.parquet",
    shell:
        "cd 3_analysis-prediction && python prep_top50_samesex_corr.py"

rule prep_years_vs_auc:
    input:
        "data/df_analysis_slim.pkl",
    output:
        "data/fig_table_gen/years_vs_auc.parquet",
    shell:
        "cd 3_analysis-prediction && python prep_years_vs_auc.py"


# ------------------------ stage 4: figures & tables ----------------------

rule fig_figure2_architecture:
    output:
        "output/figure_conceptual.pdf",
        "output/figure_conceptual.png",
    shell:
        "cd 4_figures-tables && python figure2_architecture.py"

rule fig_figure_alpaca_vs_mf_individualauc_qctrl_diff_nq:
    output:
        "output/figure_alpaca_vs_mf_individualauc_qctrl_diff_nq.pdf",
        "output/figure_alpaca_vs_mf_individualauc_qctrl_diff_nq.png",
    shell:
        "cd 4_figures-tables && Rscript figure_alpaca_vs_mf_individualauc_qctrl_diff_nq.R"

rule fig_figure_auc_by_agreement:
    output:
        "output/figure_auc_by_agreement.pdf",
        "output/figure_auc_by_agreement.png",
    shell:
        "cd 4_figures-tables && Rscript figure_auc_by_agreement.R"

rule fig_figure_counterfactual:
    output:
        "output/figure_counterfactual.pdf",
        "output/figure_counterfactual.png",
    shell:
        "cd 4_figures-tables && python figure_counterfactual.py"

rule fig_figure_demographic_cv:
    output:
        "output/figure_demographic_cv.pdf",
        "output/figure_demographic_cv.png",
    shell:
        "cd 4_figures-tables && Rscript figure_demographic_cv.R"

rule fig_figure_distance_nearest_year_dummy:
    input:
        "data/fig_table_gen/distance_nearest_year.parquet",
    output:
        "output/figure_distance_nearest_year_dummy.pdf",
        "output/figure_distance_nearest_year_dummy.png",
    shell:
        "cd 4_figures-tables && python figure_distance_nearest_year_dummy.py"

rule fig_figure_embeddings:
    input:
        "data/fig_table_gen/embeddings_question.parquet",
        "data/fig_table_gen/embeddings_respondent.parquet",
        "data/fig_table_gen/embeddings_period.parquet",
    output:
        "output/figure_embeddings.pdf",
        "output/figure_embeddings.png",
    shell:
        "cd 4_figures-tables && python figure_embeddings.py"

rule fig_figure_groupvar_between:
    output:
        "output/figure_groupvar_between.pdf",
        "output/figure_groupvar_between.png",
    shell:
        "cd 4_figures-tables && Rscript figure_groupvar_between.R"

rule fig_figure_groupvar_within:
    output:
        "output/figure_groupvar_within.pdf",
        "output/figure_groupvar_within.png",
    shell:
        "cd 4_figures-tables && Rscript figure_groupvar_within.R"

rule fig_figure_individualauc_year_continuous_qctrl_diff_nq:
    output:
        "output/figure_individualauc_year_continuous_qctrl_diff_nq.pdf",
        "output/figure_individualauc_year_continuous_qctrl_diff_nq.png",
    shell:
        "cd 4_figures-tables && Rscript figure_individualauc_year_continuous_qctrl_diff_nq.R"

rule fig_figure_missing_mech:
    output:
        "output/figure_missing_mech.pdf",
        "output/figure_missing_mech.png",
    shell:
        "cd 4_figures-tables && Rscript figure_missing_mech.R"

rule fig_figure_missing_prop:
    output:
        "output/figure_missing_prop.pdf",
        "output/figure_missing_prop.png",
    shell:
        "cd 4_figures-tables && Rscript figure_missing_prop.R"

rule fig_figure_modelperformance:
    output:
        "output/figure_modelperformance.pdf",
        "output/figure_modelperformance.png",
    shell:
        "cd 4_figures-tables && python figure_modelperformance.py"

rule fig_figure_modelperformance_moduleremoved:
    output:
        "output/figure_modelperformance_moduleremoved.pdf",
        "output/figure_modelperformance_moduleremoved.png",
    shell:
        "cd 4_figures-tables && python figure_modelperformance_moduleremoved.py"

rule fig_figure_n_questions_vs_auc:
    input:
        "data/fig_table_gen/auc_by_nq.parquet",
    output:
        "output/figure_n_questions_vs_auc.pdf",
        "output/figure_n_questions_vs_auc.png",
    shell:
        "cd 4_figures-tables && python figure_n_questions_vs_auc.py"

rule fig_figure_opinionauc:
    output:
        "output/figure_opinionauc.pdf",
        "output/figure_opinionauc.png",
    shell:
        "cd 4_figures-tables && Rscript figure_opinionauc.R"

rule fig_figure_polviews_rank_scatter:
    input:
        "data/fig_table_gen/polviews_rank.parquet",
    output:
        "output/figure_polviews_rank_scatter.pdf",
        "output/figure_polviews_rank_scatter.png",
    shell:
        "cd 4_figures-tables && python figure_polviews_rank_scatter.py"

rule fig_figure_regime_schematic:
    output:
        "output/figure_regime_schematic.pdf",
        "output/figure_regime_schematic.png",
    shell:
        "cd 4_figures-tables && python figure_regime_schematic.py"

rule fig_figure_response_category:
    output:
        "output/figure_response_category.pdf",
        "output/figure_response_category.png",
    shell:
        "cd 4_figures-tables && Rscript figure_response_category.R"

rule fig_figure_retro_interp_forecast:
    input:
        "data/fig_table_gen/retro_interp_forecast_varyear.parquet",
    output:
        "output/figure_retro_interp_forecast.pdf",
        "output/figure_retro_interp_forecast.png",
    shell:
        "cd 4_figures-tables && python figure_retro_interp_forecast.py"

rule fig_figure_roper_by_existence:
    input:
        "data/fig_table_gen/roper_by_existence.parquet",
    output:
        "output/figure_roper_by_existence.pdf",
        "output/figure_roper_by_existence.png",
    shell:
        "cd 4_figures-tables && python figure_roper_by_existence.py"

rule fig_figure_samesex_exclude:
    output:
        "output/figure_samesex_exclude.pdf",
        "output/figure_samesex_exclude.png",
    shell:
        "cd 4_figures-tables && Rscript figure_samesex_exclude.R"

rule fig_figure_sorting_by_year_dummy:
    output:
        "output/figure_sorting_by_year_dummy.pdf",
        "output/figure_sorting_by_year_dummy.png",
    shell:
        "cd 4_figures-tables && Rscript figure_sorting_by_year_dummy.R"

rule fig_figure_varselection:
    output:
        "output/figure_varselection.pdf",
        "output/figure_varselection.png",
    shell:
        "cd 4_figures-tables && python figure_varselection.py"

rule fig_figure_years_vs_auc:
    input:
        "data/fig_table_gen/years_vs_auc.parquet",
    output:
        "output/figure_years_vs_auc.pdf",
        "output/figure_years_vs_auc.png",
    shell:
        "cd 4_figures-tables && python figure_years_vs_auc.py"

rule fig_table_accuracythresholds:
    output:
        "output/table_accuracy_thresholds.tex",
    shell:
        "cd 4_figures-tables && python table_accuracythresholds.py"

rule fig_table_alpaca_examples:
    input:
        "data/fig_table_gen/alpaca_examples.parquet",
    output:
        "output/table_alpaca_text_examples.tex",
    shell:
        "cd 4_figures-tables && python table_alpaca_examples.py"

rule fig_table_binarization:
    output:
        "output/table_binary_transformation.tex",
    shell:
        "cd 4_figures-tables && python table_binarization.py"

rule fig_table_counterfactual_roper_surveys:
    input:
        "data/fig_table_gen/counterfactual_roper_surveys.parquet",
    output:
        "output/table_counterfactual_roper_surveys.tex",
    shell:
        "cd 4_figures-tables && python table_counterfactual_roper_surveys.py"

rule fig_table_featureimportance:
    input:
        "data/fig_table_gen/featureimportance_permutation.parquet",
    output:
        "output/table_featureimportance.tex",
    shell:
        "cd 4_figures-tables && python table_featureimportance.py"

rule fig_table_frontier_llm:
    input:
        "data/fig_table_gen/frontier_llm_metrics.csv",
        "data/fig_table_gen/modelcomparison_metrics.csv",
    output:
        "output/table_frontier_llm.tex",
    shell:
        "cd 4_figures-tables && python table_frontier_llm.py"

rule fig_table_hybrid_mf:
    input:
        "data/fig_table_gen/concat_mf.parquet",
    output:
        "output/table_hybrid_mf.tex",
    shell:
        "cd 4_figures-tables && python table_hybrid_mf.py"

rule fig_table_modelcomparison:
    output:
        "output/table_modelcomparison.tex",
    shell:
        "cd 4_figures-tables && python table_modelcomparison.py"

rule fig_table_perfbygroup:
    output:
        "output/table_perfbygroup.tex",
    shell:
        "cd 4_figures-tables && python table_perfbygroup.py"

rule fig_table_polviews_rank:
    input:
        "data/fig_table_gen/polviews_rank.parquet",
    output:
        "output/table_polviews_rank.tex",
    shell:
        "cd 4_figures-tables && python table_polviews_rank.py"

rule fig_table_questiontype:
    output:
        "output/table_question_distribution.tex",
    shell:
        "cd 4_figures-tables && python table_questiontype.py"

rule fig_table_retro_distance_metrics:
    input:
        "data/fig_table_gen/retro_distance_metrics.csv",
    output:
        "output/table_retro_distance_metrics.tex",
    shell:
        "cd 4_figures-tables && python table_retro_distance_metrics.py"

rule fig_table_retro_interp_forecast:
    input:
        "data/fig_table_gen/retro_interp_forecast_metrics.csv",
    output:
        "output/table_retro_interp_forecast.tex",
    shell:
        "cd 4_figures-tables && python table_retro_interp_forecast.py"

rule fig_table_samesexframing:
    input:
        "data/fig_table_gen/samesexframing_paired_individual.parquet",
        "data/gss_train_vars_corrected_binarized_again.parquet",
    output:
        "output/table_samesex_framing.tex",
    shell:
        "cd 4_figures-tables && python table_samesexframing.py"

rule fig_table_similar_questions:
    input:
        "data/fig_table_gen/similar_questions.parquet",
    output:
        "output/table_similar_questions.tex",
    shell:
        "cd 4_figures-tables && python table_similar_questions.py"

rule fig_table_subgroup_delta_alpaca_vs_mf:
    output:
        "output/table_subgroup_delta_imputation.tex",
        "output/table_subgroup_delta_retrodiction.tex",
    shell:
        "cd 4_figures-tables && python table_subgroup_delta_alpaca_vs_mf.py"

rule fig_table_top20_modules:
    output:
        "output/table_top20_modules.tex",
    shell:
        "cd 4_figures-tables && python table_top20_modules.py"

rule fig_table_top50_samesex_corr:
    output:
        "output/table_samesex_50.tex",
    shell:
        "cd 4_figures-tables && python table_top50_samesex_corr.py"

rule fig_figure_ft_effect:
    output:
        "output/figure_ft_effect.pdf",
        "output/figure_ft_effect.png",
    shell:
        "cd 5_finetuning-llm && Rscript figure_ft_effect.R"


# ------------------------------- targets ---------------------------------

rule analysis:
    input:
        "data/fig_table_gen/_period_controls.parquet",
        "data/fig_table_gen/accuracythresholds_alpaca.csv",
        "data/fig_table_gen/alpaca_examples.parquet",
        "data/fig_table_gen/alpaca_vs_mf_groupauc.csv",
        "data/fig_table_gen/alpaca_vs_mf_individualauc_per_respondent.parquet",
        "data/fig_table_gen/auc_by_agreement.parquet",
        "data/fig_table_gen/auc_by_nq.parquet",
        "data/fig_table_gen/concat_mf.parquet",
        "data/fig_table_gen/counterfactual_pop_mean.parquet",
        "data/fig_table_gen/counterfactual_variable_meta.parquet",
        "data/fig_table_gen/demographic_cv.parquet",
        "data/fig_table_gen/distance_nearest_year.parquet",
        "data/fig_table_gen/embeddings_period.parquet",
        "data/fig_table_gen/embeddings_question.parquet",
        "data/fig_table_gen/embeddings_respondent.parquet",
        "data/fig_table_gen/featureimportance_permutation.parquet",
        "data/fig_table_gen/fig_groupvar_between.parquet",
        "data/fig_table_gen/fig_groupvar_between_summary.csv",
        "data/fig_table_gen/fig_groupvar_within.parquet",
        "data/fig_table_gen/fig_groupvar_within_summary.csv",
        "data/fig_table_gen/frontier_llm_metrics.csv",
        "data/fig_table_gen/individualauc_per_respondent.parquet",
        "data/fig_table_gen/missing_mech_auc.parquet",
        "data/fig_table_gen/missing_prop_auc.parquet",
        "data/fig_table_gen/modelcomparison_metrics.csv",
        "data/fig_table_gen/modelperformance_roc_auc.csv",
        "data/fig_table_gen/modelperformance_roc_downsampled.parquet",
        "data/fig_table_gen/modelperformance_scatter.parquet",
        "data/fig_table_gen/opinionauc_per_varyear.parquet",
        "data/fig_table_gen/perfbygroup_metrics.csv",
        "data/fig_table_gen/polviews_rank.parquet",
        "data/fig_table_gen/response_category_auc.parquet",
        "data/fig_table_gen/retro_distance_metrics.csv",
        "data/fig_table_gen/retro_interp_forecast_metrics.csv",
        "data/fig_table_gen/retro_interp_forecast_varyear.parquet",
        "data/fig_table_gen/samesex_exclude.parquet",
        "data/fig_table_gen/samesexframing_paired_individual.parquet",
        "data/fig_table_gen/similar_questions.parquet",
        "data/fig_table_gen/sorting_by_year.parquet",
        "data/fig_table_gen/top20_modules.parquet",
        "data/fig_table_gen/top50_samesex_corr.parquet",
        "data/fig_table_gen/years_vs_auc.parquet",

rule figures:
    input:
        "output/figure_alpaca_vs_mf_individualauc_qctrl_diff_nq.pdf",
        "output/figure_alpaca_vs_mf_individualauc_qctrl_diff_nq.png",
        "output/figure_auc_by_agreement.pdf",
        "output/figure_auc_by_agreement.png",
        "output/figure_conceptual.pdf",
        "output/figure_conceptual.png",
        "output/figure_counterfactual.pdf",
        "output/figure_counterfactual.png",
        "output/figure_demographic_cv.pdf",
        "output/figure_demographic_cv.png",
        "output/figure_distance_nearest_year_dummy.pdf",
        "output/figure_distance_nearest_year_dummy.png",
        "output/figure_embeddings.pdf",
        "output/figure_embeddings.png",
        "output/figure_ft_effect.pdf",
        "output/figure_ft_effect.png",
        "output/figure_groupvar_between.pdf",
        "output/figure_groupvar_between.png",
        "output/figure_groupvar_within.pdf",
        "output/figure_groupvar_within.png",
        "output/figure_individualauc_year_continuous_qctrl_diff_nq.pdf",
        "output/figure_individualauc_year_continuous_qctrl_diff_nq.png",
        "output/figure_missing_mech.pdf",
        "output/figure_missing_mech.png",
        "output/figure_missing_prop.pdf",
        "output/figure_missing_prop.png",
        "output/figure_modelperformance.pdf",
        "output/figure_modelperformance.png",
        "output/figure_modelperformance_moduleremoved.pdf",
        "output/figure_modelperformance_moduleremoved.png",
        "output/figure_n_questions_vs_auc.pdf",
        "output/figure_n_questions_vs_auc.png",
        "output/figure_opinionauc.pdf",
        "output/figure_opinionauc.png",
        "output/figure_polviews_rank_scatter.pdf",
        "output/figure_polviews_rank_scatter.png",
        "output/figure_regime_schematic.pdf",
        "output/figure_regime_schematic.png",
        "output/figure_response_category.pdf",
        "output/figure_response_category.png",
        "output/figure_retro_interp_forecast.pdf",
        "output/figure_retro_interp_forecast.png",
        "output/figure_roper_by_existence.pdf",
        "output/figure_roper_by_existence.png",
        "output/figure_samesex_exclude.pdf",
        "output/figure_samesex_exclude.png",
        "output/figure_sorting_by_year_dummy.pdf",
        "output/figure_sorting_by_year_dummy.png",
        "output/figure_varselection.pdf",
        "output/figure_varselection.png",
        "output/figure_years_vs_auc.pdf",
        "output/figure_years_vs_auc.png",
        "output/table_accuracy_thresholds.tex",
        "output/table_alpaca_text_examples.tex",
        "output/table_binary_transformation.tex",
        "output/table_counterfactual_roper_surveys.tex",
        "output/table_featureimportance.tex",
        "output/table_frontier_llm.tex",
        "output/table_hybrid_mf.tex",
        "output/table_modelcomparison.tex",
        "output/table_perfbygroup.tex",
        "output/table_polviews_rank.tex",
        "output/table_question_distribution.tex",
        "output/table_retro_distance_metrics.tex",
        "output/table_retro_interp_forecast.tex",
        "output/table_samesex_50.tex",
        "output/table_samesex_framing.tex",
        "output/table_similar_questions.tex",
        "output/table_subgroup_delta_imputation.tex",
        "output/table_subgroup_delta_retrodiction.tex",
        "output/table_top20_modules.tex",

rule all:
    input:
        rules.analysis.input,
        rules.figures.input,
    default_target: True
