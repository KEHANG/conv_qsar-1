# GRAPHFP
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21/tox21_ahr.cfg  
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21/tox21_ar.cfg   
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21/tox21_ar-lbd.cfg     
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21/tox21_er.cfg      
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21/tox21_mmp.cfg
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21/tox21_aromatase.cfg  
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21/tox21_er-lbd.cfg  
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21/tox21_p53.cfg
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21/tox21_are.cfg  
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21/tox21_atad5.cfg      
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21/tox21_hse.cfg     
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21/tox21_ppar-gamma.cfg

# Testing combined CV on test data (leaderboard)
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21/tox21_ahr.cfg  
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21/tox21_ar.cfg   
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21/tox21_ar-lbd.cfg     
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21/tox21_er.cfg      
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21/tox21_mmp.cfg
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21/tox21_aromatase.cfg  
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21/tox21_er-lbd.cfg  
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21/tox21_p53.cfg
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21/tox21_are.cfg  
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21/tox21_atad5.cfg      
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21/tox21_hse.cfg     
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21/tox21_ppar-gamma.cfg

# Test combined CV on evaluation data (without using extra leaderboard for training)
python conv_qsar/scripts/consensus_eval_from_CV.py conv_qsar/inputs/tox21/tox21_ahr.cfg  
python conv_qsar/scripts/consensus_eval_from_CV.py conv_qsar/inputs/tox21/tox21_ar.cfg   
python conv_qsar/scripts/consensus_eval_from_CV.py conv_qsar/inputs/tox21/tox21_ar-lbd.cfg     
python conv_qsar/scripts/consensus_eval_from_CV.py conv_qsar/inputs/tox21/tox21_er.cfg      
python conv_qsar/scripts/consensus_eval_from_CV.py conv_qsar/inputs/tox21/tox21_mmp.cfg
python conv_qsar/scripts/consensus_eval_from_CV.py conv_qsar/inputs/tox21/tox21_aromatase.cfg  
python conv_qsar/scripts/consensus_eval_from_CV.py conv_qsar/inputs/tox21/tox21_er-lbd.cfg  
python conv_qsar/scripts/consensus_eval_from_CV.py conv_qsar/inputs/tox21/tox21_p53.cfg
python conv_qsar/scripts/consensus_eval_from_CV.py conv_qsar/inputs/tox21/tox21_are.cfg  
python conv_qsar/scripts/consensus_eval_from_CV.py conv_qsar/inputs/tox21/tox21_atad5.cfg      
python conv_qsar/scripts/consensus_eval_from_CV.py conv_qsar/inputs/tox21/tox21_hse.cfg     
python conv_qsar/scripts/consensus_eval_from_CV.py conv_qsar/inputs/tox21/tox21_ppar-gamma.cfg

# Training on training and test (leaderboard) data, pre-evaluation
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_traintest/tox21_ahr.cfg  
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_traintest/tox21_ar.cfg   
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_traintest/tox21_ar-lbd.cfg     
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_traintest/tox21_er.cfg      
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_traintest/tox21_mmp.cfg
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_traintest/tox21_aromatase.cfg  
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_traintest/tox21_er-lbd.cfg  
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_traintest/tox21_p53.cfg
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_traintest/tox21_are.cfg  
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_traintest/tox21_atad5.cfg      
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_traintest/tox21_hse.cfg     
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_traintest/tox21_ppar-gamma.cfg


# Testing combined train-test models on evaluation dataset
python conv_qsar/scripts/consensus_eval_from_CV.py conv_qsar/inputs/tox21_traintest/tox21_ahr.cfg  
python conv_qsar/scripts/consensus_eval_from_CV.py conv_qsar/inputs/tox21_traintest/tox21_ar.cfg   
python conv_qsar/scripts/consensus_eval_from_CV.py conv_qsar/inputs/tox21_traintest/tox21_ar-lbd.cfg     
python conv_qsar/scripts/consensus_eval_from_CV.py conv_qsar/inputs/tox21_traintest/tox21_er.cfg      
python conv_qsar/scripts/consensus_eval_from_CV.py conv_qsar/inputs/tox21_traintest/tox21_mmp.cfg
python conv_qsar/scripts/consensus_eval_from_CV.py conv_qsar/inputs/tox21_traintest/tox21_aromatase.cfg  
python conv_qsar/scripts/consensus_eval_from_CV.py conv_qsar/inputs/tox21_traintest/tox21_er-lbd.cfg  
python conv_qsar/scripts/consensus_eval_from_CV.py conv_qsar/inputs/tox21_traintest/tox21_p53.cfg
python conv_qsar/scripts/consensus_eval_from_CV.py conv_qsar/inputs/tox21_traintest/tox21_are.cfg  
python conv_qsar/scripts/consensus_eval_from_CV.py conv_qsar/inputs/tox21_traintest/tox21_atad5.cfg      
python conv_qsar/scripts/consensus_eval_from_CV.py conv_qsar/inputs/tox21_traintest/tox21_hse.cfg     
python conv_qsar/scripts/consensus_eval_from_CV.py conv_qsar/inputs/tox21_traintest/tox21_ppar-gamma.cfg

# Multitask
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21/tox21_all.cfg

# MORGAN FP (512) ONES
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan/tox21_ahr.cfg  
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan/tox21_ar-lbd.cfg     
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan/tox21_er.cfg      
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan/tox21_mmp.cfg
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan/tox21_ar.cfg   
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan/tox21_aromatase.cfg  
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan/tox21_er-lbd.cfg  
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan/tox21_p53.cfg
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan/tox21_are.cfg  
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan/tox21_atad5.cfg      
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan/tox21_hse.cfg     
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan/tox21_ppar-gamma.cfg
# MORGAN FP (512) ONES

python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan/tox21_ahr.cfg  
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan/tox21_ar-lbd.cfg     
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan/tox21_er.cfg      
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan/tox21_mmp.cfg
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan/tox21_ar.cfg   
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan/tox21_aromatase.cfg  
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan/tox21_er-lbd.cfg  
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan/tox21_p53.cfg
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan/tox21_are.cfg  
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan/tox21_atad5.cfg      
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan/tox21_hse.cfg     
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan/tox21_ppar-gamma.cfg


# MORGAN WITH RADIUS 2
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan2/tox21_ahr.cfg  
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan2/tox21_ar-lbd.cfg     
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan2/tox21_er.cfg      
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan2/tox21_mmp.cfg
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan2/tox21_ar.cfg   
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan2/tox21_aromatase.cfg  
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan2/tox21_er-lbd.cfg  
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan2/tox21_p53.cfg
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan2/tox21_are.cfg  
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan2/tox21_atad5.cfg      
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan2/tox21_hse.cfg     
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan2/tox21_ppar-gamma.cfg


python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan2/tox21_ahr.cfg  
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan2/tox21_ar-lbd.cfg     
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan2/tox21_er.cfg      
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan2/tox21_mmp.cfg
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan2/tox21_ar.cfg   
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan2/tox21_aromatase.cfg  
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan2/tox21_er-lbd.cfg  
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan2/tox21_p53.cfg
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan2/tox21_are.cfg  
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan2/tox21_atad5.cfg      
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan2/tox21_hse.cfg     
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan2/tox21_ppar-gamma.cfg

