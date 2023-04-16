
import logging
import argparse

from train_util import set_seed, train_diff


def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2023,
                    help="random seed for initialization")

    parser.add_argument(
        "--data_file",
        default='proc_data/mimic/mimic1782_train.npy',
        type=str,
        help="Path to the file of preprocessed EHRs data",
    )

    parser.add_argument("--check_steps", default=5000, type=int,
                    help="the interval for printing the training loss, *batch")
    parser.add_argument("--num_epochs", default=5000, type=int,
                    help="number of epochs for model training")
    parser.add_argument("--batch_size", default=1024, type=int,
                    help="batch size")    
    parser.add_argument("--if_shuffle", default=True, type=bool,
                    help="parameter for the dataloader")    
    parser.add_argument("--if_drop_last", default=True, type=bool,
                    help="parameter for the dataloader")  
    parser.add_argument("--device", default="cuda:0", type=str,
                    help="device assigned for modeling")    

    parser.add_argument("--ehr_dim", default=1782, type=int,
                    help="data dimension of EHR data") 
    parser.add_argument("--time_dim", default=384, type=int,
                    help="dimension of time embedding") 
    parser.add_argument("--mlp_dims", nargs='+', default=[1024, 384, 384, 384, 1024], type=int,
                    help="hidden dims for the mlp backbone") 

    
    parser.add_argument("--sigma_data", default=0.14, type=float,
                    help="init parameters for sigma_data") 
    parser.add_argument("--p_mean", default=-1.2, type=float,
                    help="init parameters for p_mean") 
    parser.add_argument("--p_std", default=1.2, type=float,
                    help="init parameters for p_std") 

    parser.add_argument("--num_sample_steps", default=32, type=int,
                    help="init parameters for number of discretized time steps") 
    parser.add_argument("--sigma_min", default=0.02, type=float,
                    help="init parameters for sigma_min") 
    parser.add_argument("--sigma_max", default=80, type=float,
                    help="init parameters for sigma_max") 
    parser.add_argument("--rho", default=7, type=float,
                    help="init parameters for rho") 

    parser.add_argument("--lr", default=3e-4, type=float,
                    help="learning_rate")  
    parser.add_argument("--warmup_steps", default=20000, type=int,
                    help="warmup portion for the 'get_linear_schedule_with_warmup'")  
    parser.add_argument("--weight_decay", default=0., type=float,
                    help="weigth decay value for the optimizer")   
    
    parser.add_argument("--eval_samples", default=41000, type=int,
                    help="number of samples wanted for evaluation") 

    args = parser.parse_args()    

    model_setting = 'sigma_data' + str(args.sigma_data) + '|' + \
                    'p_mean' + str(args.p_mean) + '|' + \
                    'p_std' + str(args.p_std) + '|' + \
                    'steps' + str(args.num_sample_steps) + '|' + \
                    'sigma_min' + str(args.sigma_min) + '|' + \
                    'sigma_max' + str(args.sigma_max) + '|' + \
                    'rho' + str(args.rho)

    args.model_setting = model_setting
    logging.basicConfig(
            # filename='logs/' + model_setting + '.log',
            filename='logs/' + 'trial.log',
            level=logging.INFO,
            filemode='w',
            format='%(name)s - %(levelname)s - %(message)s'
                )

    set_seed(args.seed)
    train_diff(args)

if __name__=='__main__':
    main()