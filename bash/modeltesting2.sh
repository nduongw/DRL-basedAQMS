CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config1-model-none-changev --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --testing true --usingmodel true --mapwidth 50 \
                                    --mapheight 400 --uncover 20 --generation 20 --coverrange 10 \
                                     --clambda 0.01 --morningv 10 --afternoonv 5 --eveningv 1

CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config2-model-none-changev --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --testing true --usingmodel true --mapwidth 50 \
                                    --mapheight 400 --uncover 20 --generation 20 --coverrange 10 \
                                     --clambda 0.01 --morningv 2 --afternoonv 5 --eveningv 20

CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config3-model-none-changev --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --testing true --usingmodel true --mapwidth 50 \
                                    --mapheight 400 --uncover 20 --generation 20 --coverrange 10 \
                                     --clambda 0.01 --morningv 20 --afternoonv 5 --eveningv 20

CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config4-model-none-changev --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --testing true --usingmodel true --mapwidth 50 \
                                    --mapheight 400 --uncover 20 --generation 20 --coverrange 10 \
                                     --clambda 0.01 --morningv 5 --afternoonv 10 --eveningv 5

CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config5-model-none-changev --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --testing true --usingmodel true --mapwidth 50 \
                                    --mapheight 400 --uncover 20 --generation 20 --coverrange 10 \
                                     --clambda 0.01 --morningv 1 --afternoonv 20 --eveningv 2

CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config6-model-none-changev --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --testing true --usingmodel true --mapwidth 50 \
                                    --mapheight 400 --uncover 20 --generation 20 --coverrange 10 \
                                     --clambda 0.01 --morningv 10 --afternoonv 5 --eveningv 5