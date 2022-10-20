# Chay thi nghiem voi thuat toan PSO-based
# CUDA_VISIBLE_DEVICES=3 python main.py --storepath testing-config9-pso-poisson --model dense \
#                                     --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
#                                     --zoom 1 --testing true --sendingpercentage 1 --pso true

# Chay thi nghiem su dung model
# CUDA_VISIBLE_DEVICES=3 python main.py --storepath testing-config1-model-none --model dense \
#                                     --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
#                                     --zoom 1 --testing true --usingmodel true --mapwidth 50 \
#                                     --mapheight 400 --uncover 3 --generation 5 --coverrange 5

# CUDA_VISIBLE_DEVICES=3 python main.py --storepath testing-config1-model-none --model dense \
#                                     --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
#                                     --zoom 1 --testing true --usingmodel true --mapwidth 50 \
#                                     --mapheight 400 --uncover 3 --generation 5 --coverrange 5 \
#                                     --morningv 10 --afternoonv 5 --eveningv 1

# Chay thi nghiem voi ti le bat ngau nhien
CUDA_VISIBLE_DEVICES=3 python main.py --storepath testing-config1-turnonmodel-poisson --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --poisson true --testing true --sendingpercentage 0.11192629615973035 --mapwidth 50 \
                                    --mapheight 400 --uncover 10 --generation 8 --coverrange 10 --clambda 0.01

CUDA_VISIBLE_DEVICES=3 python main.py --storepath testing-config2-turnonmodel-poisson --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --poisson true --testing true --sendingpercentage 0.07095640512811884 --mapwidth 50 \
                                    --mapheight 400 --uncover 20 --generation 8 --coverrange 10 --clambda 0.01

# CUDA_VISIBLE_DEVICES=3 python main.py --storepath testing-config1-mae-turnonmodel-none --model dense \
#                                     --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
#                                     --zoom 1 --testing true --sendingpercentage 0.40665152812686006 --mapwidth 20 \
#                                     --mapheight 400 --uncover 2 --generation 4 --coverrange 5 \
#                                     --morningv 20 --afternoonv 5 --eveningv 20