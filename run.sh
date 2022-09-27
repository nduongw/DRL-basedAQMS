# Chay thi nghiem voi thuat toan PSO-based
CUDA_VISIBLE_DEVICES=3 python main.py --storepath testing-config9-pso-poisson --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --testing true --sendingpercentage 1 --pso true

# Chay thi nghiem su dung model
CUDA_VISIBLE_DEVICES=3 python main.py --storepath testing-config9-model-poisson --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --testing true --usingmodel true

# Chay thi nghiem voi ti le bat ngau nhien
CUDA_VISIBLE_DEVICES=3 python main.py --storepath testing-config9-turnonmodel-poisson --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --testing true --sendingpercentage 0.12169
