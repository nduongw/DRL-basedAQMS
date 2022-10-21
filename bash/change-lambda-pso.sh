CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config1-pso-poisson-lambda --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --poisson true --testing true --sendingpercentage 1 --pso true --mapwidth 50 \
                                    --mapheight 400 --uncover 20 --generation 8 --coverrange 10 \
                                     --clambda 0.005 --velocity 10

CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config2-pso-poisson-lambda1 --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --poisson true --testing true --sendingpercentage 1 --pso true --mapwidth 50 \
                                    --mapheight 400 --uncover 20 --generation 8 --coverrange 10 \
                                     --clambda 0.01 --velocity 10

CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config3-pso-poisson-lambda --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --poisson true --testing true --sendingpercentage 1 --pso true --mapwidth 50 \
                                    --mapheight 400 --uncover 20 --generation 8 --coverrange 10 \
                                     --clambda 0.015 --velocity 10 

CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config4-pso-poisson-lambda1 --model dense
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 
                                    --zoom 1 --poisson true --testing true --sendingpercentage 1 --pso true --mapwidth 50 
                                    --mapheight 400 --uncover 20 --generation 8 --coverrange 10 
                                     --clambda 0.02 --velocity 10

CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config5-pso-poisson-lambda --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --poisson true --testing true --sendingpercentage 1 --pso true --mapwidth 50 \
                                    --mapheight 400 --uncover 20 --generation 8 --coverrange 10 \
                                     --clambda 0.025 --velocity 10 

CUDA_VISIBLE_DEVICES=1 python main.py --storepath testing-config6-pso-poisson-lambda1 --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --poisson true --testing true --sendingpercentage 1 --pso true --mapwidth 50 \
                                    --mapheight 400 --uncover 20 --generation 8 --coverrange 10 \
                                     --clambda 0.03 --velocity 10