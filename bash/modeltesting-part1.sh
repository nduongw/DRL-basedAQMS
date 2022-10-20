CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config1-model-none --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --testing true --usingmodel true --mapwidth 50 \
                                    --mapheight 400 --uncover 10 --generation 10 --coverrange 10 --velocity 10

CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config2-model-none --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --testing true --usingmodel true --mapwidth 50 \
                                    --mapheight 400 --uncover 20 --generation 10 --coverrange 10 --velocity 10

CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config3-model-none --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --testing true --usingmodel true --mapwidth 50 \
                                    --mapheight 400 --uncover 20 --generation 20 --coverrange 8 --velocity 15

CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config4-model-none --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --testing true --usingmodel true --mapwidth 50 \
                                    --mapheight 400 --uncover 5 --generation 15 --coverrange 10 --velocity 20

CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config5-model-none --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --testing true --usingmodel true --mapwidth 50 \
                                    --mapheight 400 --uncover 5 --generation 15 --coverrange 20 --velocity 20

CUDA_VISIBLE_DEVICES=0 python main.py --storepath testing-config6-model-none --model dense \
                                    --modelpath dense-15t9-15h46-r5-zoom1 --rewardfunc ver5 \
                                    --zoom 1 --testing true --usingmodel true --mapwidth 50 \
                                    --mapheight 400 --uncover 15 --generation 20 --coverrange 3 --velocity 20