# CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config1-fixed-part1-coverrange --model dense \
#                                     --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
#                                     --zoom 1 --testing true --mapwidth 50 --sendingpercentage 0.1004\
#                                     --mapheight 400 --uncover 20 --generation 8 --coverrange 8 \
#                                     --clambda 0.01 --velocity 10

# CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config2-fixed-part1-coverrange --model dense \
#                                     --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
#                                     --zoom 1 --testing true --mapwidth 50 --sendingpercentage 0.2709\
#                                     --mapheight 400 --uncover 20 --generation 8 --coverrange 3 \
#                                     --clambda 0.01 --velocity 10

CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config3-fixed-part1-coverrange --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --testing true --mapwidth 50 --sendingpercentage 0.1427 \
                                    --mapheight 400 --uncover 20 --generation 8 --coverrange 4 \
                                    --clambda 0.01 --velocity 10 

# CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config4-fixed-part1-coverrange --model dense \
#                                     --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
#                                     --zoom 1 --testing true --mapwidth 50 --sendingpercentage 0.0981 \
#                                     --mapheight 400 --uncover 20 --generation 8 --coverrange 5 \
#                                     --clambda 0.01 --velocity 10

# CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config5-fixed-part1-coverrange --model dense \
#                                     --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
#                                     --zoom 1 --testing true --mapwidth 50 --sendingpercentage 0.1116 \
#                                     --mapheight 400 --uncover 20 --generation 8 --coverrange 6 \
#                                     --clambda 0.01 --velocity 10 

# CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config6-fixed-part1-coverrange --model dense \
#                                     --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
#                                     --zoom 1 --testing true --mapwidth 50 --sendingpercentage 0.1198 \
#                                     --mapheight 400 --uncover 20 --generation 8 --coverrange 7 \
#                                     --clambda 0.01 --velocity 10