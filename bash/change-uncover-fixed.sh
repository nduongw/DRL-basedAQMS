CUDA_VISIBLE_DEVICES=1 python main.py --storepath testing-config1-fixed-poisson-uncover --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --poisson true --testing true --sendingpercentage 0.1822  --mapwidth 50 \
                                    --mapheight 400 --uncover 5 --generation 8 --coverrange 10 \
                                     --clambda 0.01 --velocity 10

CUDA_VISIBLE_DEVICES=1 python main.py --storepath testing-config2-fixed-poisson-uncover --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --poisson true --testing true --sendingpercentage 0.112  --mapwidth 50 \
                                    --mapheight 400 --uncover 10 --generation 8 --coverrange 10 \
                                     --clambda 0.01 --velocity 10

CUDA_VISIBLE_DEVICES=1 python main.py --storepath testing-config3-fixed-poisson-uncover --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --poisson true --testing true --sendingpercentage 0.0849  --mapwidth 50 \
                                    --mapheight 400 --uncover 15 --generation 8 --coverrange 10 \
                                     --clambda 0.01 --velocity 10 

CUDA_VISIBLE_DEVICES=1 python main.py --storepath testing-config4-fixed-poisson-uncover --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --poisson true --testing true --sendingpercentage 0.071  --mapwidth 50 \
                                    --mapheight 400 --uncover 20 --generation 8 --coverrange 10 \
                                     --clambda 0.01 --velocity 10

CUDA_VISIBLE_DEVICES=1 python main.py --storepath testing-config5-fixed-poisson-uncover --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --poisson true --testing true --sendingpercentage 0.0619  --mapwidth 50 \
                                    --mapheight 400 --uncover 25 --generation 8 --coverrange 10 \
                                     --clambda 0.01 --velocity 10 

CUDA_VISIBLE_DEVICES=1 python main.py --storepath testing-config6-fixed-poisson-uncover --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --poisson true --testing true --sendingpercentage 0.0555  --mapwidth 50 \
                                    --mapheight 400 --uncover 30 --generation 8 --coverrange 10 \
                                     --clambda 0.01 --velocity 10