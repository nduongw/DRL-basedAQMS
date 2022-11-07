# Chay thi nghiem voi thuat toan PSO-based
CUDA_VISIBLE_DEVICES=2 python main.py --storepath testing-config1-pso-poisson-changev-new --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --testing true --sendingpercentage 1 --pso true \
                                    --mapwidth 50 --mapheight 400 --uncover 20 --generation 8 \
                                    --coverrange 10 --clambda 0.01 --poisson true \
                                    --morningv 10 --afternoonv 5 --eveningv 1

CUDA_VISIBLE_DEVICES=2 python main.py --storepath testing-config2-pso-poisson-changev-new --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --testing true --sendingpercentage 1 --pso true \
                                    --mapwidth 50 --mapheight 400 --uncover 20 --generation 8 \
                                    --coverrange 10 --clambda 0.01 --poisson true \
                                    --morningv 10 --afternoonv 15 --eveningv 20

CUDA_VISIBLE_DEVICES=2 python main.py --storepath testing-config3-pso-poisson-changev-new --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --testing true --sendingpercentage 1 --pso true \
                                    --mapwidth 50 --mapheight 400 --uncover 20 --generation 8 \
                                    --coverrange 10 --clambda 0.01 --poisson true \
                                    --morningv 5 --afternoonv 10 --eveningv 20

CUDA_VISIBLE_DEVICES=2 python main.py --storepath testing-config4-pso-poisson-changev-new --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --testing true --sendingpercentage 1 --pso true \
                                    --mapwidth 50 --mapheight 400 --uncover 20 --generation 8 \
                                    --coverrange 10 --clambda 0.01 --poisson true \
                                    --morningv 5 --afternoonv 10 --eveningv 5

CUDA_VISIBLE_DEVICES=2 python main.py --storepath testing-config5-pso-poisson-changev-new --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --testing true --sendingpercentage 1 --pso true \
                                    --mapwidth 50 --mapheight 400 --uncover 20 --generation 8 \
                                    --coverrange 10 --clambda 0.01 --poisson true \
                                    --morningv 5 --afternoonv 5 --eveningv 10

CUDA_VISIBLE_DEVICES=2 python main.py --storepath testing-config6-pso-poisson-changev-new --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --testing true --sendingpercentage 1 --pso true \
                                    --mapwidth 50 --mapheight 400 --uncover 20 --generation 8 \
                                    --coverrange 10 --clambda 0.01 --poisson true \
                                    --morningv 1 --afternoonv 10 --eveningv 1

# Chay thi nghiem su dung model
# CUDA_VISIBLE_DEVICES=3 python main.py --storepath testing-config1-model-poisson --model dense \
#                                     --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
#                                     --zoom 1 --poisson true --testing true --usingmodel true --mapwidth 50 \
#                                     --mapheight 400 --uncover 10 --generation 8 --coverrange 10 \
#                                      --clambda 0.01

# CUDA_VISIBLE_DEVICES=3 python main.py --storepath testing-config2-model-poisson --model dense \
#                                     --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
#                                     --zoom 1 --poisson true --testing true --usingmodel true --mapwidth 50 \
#                                     --mapheight 400 --uncover 20 --generation 8 --coverrange 10 \
#                                      --clambda 0.01

# CUDA_VISIBLE_DEVICES=3 python main.py --storepath testing-config1-mae-model-none --model dense \
#                                     --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
#                                     --zoom 1 --testing true --usingmodel true --mapwidth 20 \
#                                     --mapheight 400 --uncover 2 --generation 4 --coverrange 5 \
#                                     --morningv 20 --afternoonv 5 --eveningv 20

# Chay thi nghiem voi ti le bat ngau nhien
# CUDA_VISIBLE_DEVICES=3 python main.py --storepath testing-config1-turnonmodel-none --model dense \
#                                     --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
#                                     --zoom 1 --testing true --sendingpercentage 0.12169
