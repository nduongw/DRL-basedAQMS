CUDA_VISIBLE_DEVICES=1 python main.py --storepath testing-config1-model-part1-changev --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --sendingpercentage 1 --usingmodel true \
                                    --mapwidth 50 --mapheight 400 --uncover 20 --generation 8 \
                                    --coverrange 10 --clambda 0.01 --testing true \
                                    --morningv 10 --afternoonv 5 --eveningv 1

CUDA_VISIBLE_DEVICES=1 python main.py --storepath testing-config2-model-part1-changev --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --sendingpercentage 1 --usingmodel true \
                                    --mapwidth 50 --mapheight 400 --uncover 20 --generation 8 \
                                    --coverrange 10 --clambda 0.01 --testing true \
                                    --morningv 10 --afternoonv 15 --eveningv 20

CUDA_VISIBLE_DEVICES=1 python main.py --storepath testing-config3-model-part1-changev --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --sendingpercentage 1 --usingmodel true \
                                    --mapwidth 50 --mapheight 400 --uncover 20 --generation 8 \
                                    --coverrange 10 --clambda 0.01 --testing true \
                                    --morningv 5 --afternoonv 10 --eveningv 20

CUDA_VISIBLE_DEVICES=1 python main.py --storepath testing-config4-model-part1-changev --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --sendingpercentage 1 --usingmodel true \
                                    --mapwidth 50 --mapheight 400 --uncover 20 --generation 8 \
                                    --coverrange 10 --clambda 0.01 --testing true \
                                    --morningv 5 --afternoonv 10 --eveningv 5

CUDA_VISIBLE_DEVICES=1 python main.py --storepath testing-config5-model-part1-changev --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --sendingpercentage 1 --usingmodel true \
                                    --mapwidth 50 --mapheight 400 --uncover 20 --generation 8 \
                                    --coverrange 10 --clambda 0.01 --testing true \
                                    --morningv 5 --afternoonv 5 --eveningv 10

CUDA_VISIBLE_DEVICES=1 python main.py --storepath testing-config6-model-part1-changev --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --sendingpercentage 1 --usingmodel true \
                                    --mapwidth 50 --mapheight 400 --uncover 20 --generation 8 \
                                    --coverrange 10 --clambda 0.01 --testing true \
                                    --morningv 1 --afternoonv 10 --eveningv 1