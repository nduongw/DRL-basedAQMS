# CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config1-fixed-poisson-lambda --model dense \
#                                     --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
#                                     --zoom 1 --poisson true --testing true --sendingpercentage 0.151 --mapwidth 50 \
#                                     --mapheight 400 --uncover 20 --generation 8 --coverrange 10 \
#                                      --clambda 0.005 --velocity 10

# CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config2-fixed-poisson-lambda --model dense \
#                                     --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
#                                     --zoom 1 --poisson true --testing true --sendingpercentage 0.071 --mapwidth 50 \
#                                     --mapheight 400 --uncover 20 --generation 8 --coverrange 10 \
#                                      --clambda 0.01 --velocity 10

# CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config3-fixed-poisson-lambda --model dense \
#                                     --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
#                                     --zoom 1 --poisson true --testing true --sendingpercentage 0.0546 --mapwidth 50 \
#                                     --mapheight 400 --uncover 20 --generation 8 --coverrange 10 \
#                                      --clambda 0.015 --velocity 10 

CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config4-fixed-poisson-lambda --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --poisson true --testing true --sendingpercentage 0.0444 --mapwidth 50 \
                                    --mapheight 400 --uncover 20 --generation 8 --coverrange 10 \
                                     --clambda 0.02 --velocity 10

# CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config5-fixed-poisson-lambda --model dense \
#                                     --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
#                                     --zoom 1 --poisson true --testing true --sendingpercentage 0.0374 --mapwidth 50 \
#                                     --mapheight 400 --uncover 20 --generation 8 --coverrange 10 \
#                                      --clambda 0.025 --velocity 10 

# CUDA_VISIBLE_DEVICES=1 python main.py --storepath testing-config6-fixed-poisson-lambda --model dense \
#                                     --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
#                                     --zoom 1 --poisson true --testing true --sendingpercentage 0.0333 --mapwidth 50 \
#                                     --mapheight 400 --uncover 20 --generation 8 --coverrange 10 \
#                                      --clambda 0.03 --velocity 10