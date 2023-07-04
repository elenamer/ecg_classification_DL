# Classification results

Using this framework, we ran a comparative study of ECG arrhythmia classification. In this folder we provide more extensive classification scores.

## Comparison of inter- and intra-patient evaluation
 
To demonstrate the difference between inter- and intra-patient evaluation, we have provided results (**test set F1 score**) for form classification on three beat-level datasets in the table 
*summary_form_evaluation_inter_vs_intra.csv*. 

These results were obtained with cross-validation. From these numbers, we can observe how the intra-patient evaluation schema gives much higher scores. Considering that intra-patient is not a realistic scenario, it demonstrates that it is an overly optimistic evaluation.

## Extensive scores

To give a more comprehensive overview of arrhythmia classification, we provide classification scores on the **train, validation and test sets**. This includes **per-class metrics (F1 and AUC)**, as well as average scores over all classes (macro average).

### Folowing are the files related to each setting

Form classification results on beat-level datasets: *summary_form_beatlevel.csv*
Form classification results on recording datasets: *summary_form_recordinglevel.csv*
Rhythm classification results on beat-level datasets: *summary_rhythm_beatlevel.csv*
Rhythm classification results on recording datasets: *summary_rhythm_recordinglevel.csv*

