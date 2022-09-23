CUDA_VISIBLE_DEVICES=3 python main.py --storepath testing-config3-pso-poisson --model dense --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 --zoom 1 --testing True --sendingpercentage 1 --pso True

CUDA_VISIBLE_DEVICES=3 python main.py --storepath testing-config3-model-poisson --model dense --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 --zoom 1 --testing True --usingmodel true