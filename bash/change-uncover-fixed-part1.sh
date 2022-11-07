CUDA_VISIBLE_DEVICES=1 python main.py --storepath testing-config1-fixed-part1-uncover --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --testing true --sendingpercentage 0.1444  --mapwidth 50 \
                                    --mapheight 400 --uncover 5 --generation 8 --coverrange 10 \
                                     --clambda 0.01 --velocity 10

CUDA_VISIBLE_DEVICES=1 python main.py --storepath testing-config2-fixed-part1-uncover --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --testing true --sendingpercentage 0.0979  --mapwidth 50 \
                                    --mapheight 400 --uncover 10 --generation 8 --coverrange 10 \
                                     --clambda 0.01 --velocity 10

CUDA_VISIBLE_DEVICES=1 python main.py --storepath testing-config3-fixed-part1-uncover --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --testing true --sendingpercentage 0.0812  --mapwidth 50 \
                                    --mapheight 400 --uncover 15 --generation 8 --coverrange 10 \
                                     --clambda 0.01 --velocity 10 

CUDA_VISIBLE_DEVICES=1 python main.py --storepath testing-config4-fixed-part1-uncover --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --testing true --sendingpercentage 0.0726  --mapwidth 50 \
                                    --mapheight 400 --uncover 20 --generation 8 --coverrange 10 \
                                     --clambda 0.01 --velocity 10

CUDA_VISIBLE_DEVICES=1 python main.py --storepath testing-config5-fixed-part1-uncover --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --testing true --sendingpercentage 0.0673  --mapwidth 50 \
                                    --mapheight 400 --uncover 25 --generation 8 --coverrange 10 \
                                     --clambda 0.01 --velocity 10 

CUDA_VISIBLE_DEVICES=1 python main.py --storepath testing-config6-fixed-part1-uncover --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --testing true --sendingpercentage 0.063  --mapwidth 50 \
                                    --mapheight 400 --uncover 30 --generation 8 --coverrange 10 \
                                     --clambda 0.01 --velocity 10