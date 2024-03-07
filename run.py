from util.solver import Solver
from tensorboard import program
from multiprocessing import Process
from util.parser import get_parser

def start_tensorboard(log_dir):
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', log_dir])
    url = tb.main()
    return url

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    solver = Solver(args)
    if args.mode == 'train':
        sub_process = Process(target=start_tensorboard,kwargs={'log_dir':solver.args.log_dir})
        sub_process.start()
        solver.train()
    elif args.mode == 'test':
        if args.resume_epoch ==0:
            solver.load_model(latest=True)
        solver.test()
    else:
        raise(Exception('Wrong Mode!'))

    