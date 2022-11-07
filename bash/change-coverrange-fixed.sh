# CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config1-model-poisson-coverrange-new --model dense \
#                                     --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
#                                     --zoom 1 --poisson true --testing true --mapwidth 50 --sendingpercentage  \
#                                     --mapheight 400 --uncover 20 --generation 8 --coverrange 2 \
#                                      --clambda 0.01 --velocity 10

CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config2-model-poisson-coverrange-new --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --poisson true --testing true --mapwidth 50 --sendingpercentage 0.2358 \
                                    --mapheight 400 --uncover 20 --generation 8 --coverrange 3 \
                                     --clambda 0.01 --velocity 10

CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config3-model-poisson-coverrange-new --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --poisson true --testing true --mapwidth 50 --sendingpercentage 0.1108 \
                                    --mapheight 400 --uncover 20 --generation 8 --coverrange 4 \
                                     --clambda 0.01 --velocity 10 

CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config4-model-poisson-coverrange-new --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --poisson true --testing true --mapwidth 50 --sendingpercentage 0.071\
                                    --mapheight 400 --uncover 20 --generation 8 --coverrange 5 \
                                     --clambda 0.01 --velocity 10

CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config5-model-poisson-coverrange-new --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --poisson true --testing true --mapwidth 50 --sendingpercentage 0.082\
                                    --mapheight 400 --uncover 20 --generation 8 --coverrange 6 \
                                     --clambda 0.01 --velocity 10 

CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config6-model-poisson-coverrange-new --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --poisson true --testing true --mapwidth 50 --sendingpercentage 0.089\
                                    --mapheight 400 --uncover 20 --generation 8 --coverrange 7 \
                                     --clambda 0.01 --velocity 10