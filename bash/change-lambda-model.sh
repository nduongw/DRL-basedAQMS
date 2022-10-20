CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config1-model-poisson-lambda --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --poisson true --testing true --usingmodel true --mapwidth 50 \
                                    --mapheight 400 --uncover 20 --generation 8 --coverrange 10 \
                                     --velocity 10 --clambda 0.005

CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config2-model-poisson-lambda --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --poisson true --testing true --usingmodel true --mapwidth 50 \
                                    --mapheight 400 --uncover 20 --generation 8 --coverrange 10 \
                                     --velocity 10 --clambda 0.01

CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config3-model-poisson-lambda --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --poisson true --testing true --usingmodel true --mapwidth 50 \
                                    --mapheight 400 --uncover 20 --generation 8 --coverrange 10 \
                                     --velocity 10 --clambda 0.015 

CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config4-model-poisson-lambda --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --poisson true --testing true --usingmodel true --mapwidth 50 \
                                    --mapheight 400 --uncover 20 --generation 8 --coverrange 10 \
                                     --velocity 10 --clambda 0.02

CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config5-model-poisson-lambda --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --poisson true --testing true --usingmodel true --mapwidth 50 \
                                    --mapheight 400 --uncover 20 --generation 8 --coverrange 10 \
                                     --velocity 10 --clambda 0.025 

CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config6-model-poisson-lambda --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --poisson true --testing true --usingmodel true --mapwidth 50 \
                                    --mapheight 400 --uncover 20 --generation 8 --coverrange 10 \
                                    --velocity 10  --clambda 0.03