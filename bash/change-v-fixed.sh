CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config1-fixed-poisson-changev-new --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --testing true --sendingpercentage 0.07 \
                                    --mapwidth 50 --mapheight 400 --uncover 20 --generation 8 \
                                    --coverrange 10 --clambda 0.01 --poisson true \
                                    --morningv 10 --afternoonv 5 --eveningv 1

CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config2-fixed-poisson-changev-new --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --testing true --sendingpercentage 0.095 \
                                    --mapwidth 50 --mapheight 400 --uncover 20 --generation 8 \
                                    --coverrange 10 --clambda 0.01 --poisson true \
                                    --morningv 10 --afternoonv 15 --eveningv 20

CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config3-fixed-poisson-changev-new --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --testing true --sendingpercentage 0.0782 \
                                    --mapwidth 50 --mapheight 400 --uncover 20 --generation 8 \
                                    --coverrange 10 --clambda 0.01 --poisson true \
                                    --morningv 15 --afternoonv 1 --eveningv 5

CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config4-fixed-poisson-changev-new --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --testing true --sendingpercentage 0.0793 \
                                    --mapwidth 50 --mapheight 400 --uncover 20 --generation 8 \
                                    --coverrange 10 --clambda 0.01 --poisson true \
                                    --morningv 15 --afternoonv 5 --eveningv 15


CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config5-fixed-poisson-changev-new --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --testing true --sendingpercentage 0.0696 \
                                    --mapwidth 50 --mapheight 400 --uncover 20 --generation 8 \
                                    --coverrange 10 --clambda 0.01 --poisson true \
                                    --morningv 5 --afternoonv 5 --eveningv 10

CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config6-fixed-poisson-changev-new --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --testing true --sendingpercentage 0.0676 \
                                    --mapwidth 50 --mapheight 400 --uncover 20 --generation 8 \
                                    --coverrange 10 --clambda 0.01 --poisson true \
                                    --morningv 1 --afternoonv 10 --eveningv 1