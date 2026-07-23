# Exemplar retrieval spot check

20 random test-split queries with their top-1 retrieved
exemplar. Rate each pair by hand: does the exemplar read as a
"similar documented case" for the query photo? (good / weak / wrong)

| # | query image | query class | exemplar | exemplar caption | rating |
|---|---|---|---|---|---|
| 1 | data/raw/codebrim/background/background__image_0001046_crop_0000002.png | no_defect | mbdd-crack-004 | Facade crack, UAV survey crop (multi-material building walls) | |
| 2 | data/raw/bd3/minor_crack/BD3_original_dataset__train__minor_crack__cls02_125.jpg | crack | mbdd-crack-004 | Facade crack, UAV survey crop (multi-material building walls) | |
| 3 | data/raw/mbdd2025/abscission/abscission__Hefei8917_0.jpg | finish_detachment | mbdd-crack-008 | Facade crack, UAV survey crop (multi-material building walls) | |
| 4 | data/raw/mbdd2025/abscission/abscission__Hefei11712_4.jpg | finish_detachment | mbdd-crack-008 | Facade crack, UAV survey crop (multi-material building walls) | |
| 5 | data/raw/insulator/pollution_flashover/pollution_flashover__train_161020v_1.jpg | insulator_damage | insu-pollution-flashover-002 | Pollution-flashover damage on a grid transmission insulator | |
| 6 | data/raw/codebrim/crack/crack__image_0001091_crop_0000002.png | crack | sdnet-cracked-001 | Cracked concrete surface patch | |
| 7 | data/raw/codebrim/background/background__image_0000877_crop_0000004.png | no_defect | sdnet-cracked-001 | Cracked concrete surface patch | |
| 8 | data/raw/sdnet2018/non_cracked/P__UP__057-217.jpg | no_defect | sdnet-cracked-001 | Cracked concrete surface patch | |
| 9 | data/raw/codebrim/background/background__image_0000314_crop_0000003.png | no_defect | sdnet-non-cracked-002 | Sound concrete surface patch (no defect) | |
| 10 | data/raw/mbdd2025/leakage/leakage__Hefei8190_0.jpg | water_damage | mbdd-crack-008 | Facade crack, UAV survey crop (multi-material building walls) | |
| 11 | data/raw/bd3/minor_crack/BD3_original_dataset__train__minor_crack__cls02_518.jpg | crack | mbdd-crack-005 | Facade crack, UAV survey crop (multi-material building walls) | |
| 12 | data/raw/bd3/minor_crack/BD3_original_dataset__train__minor_crack__cls02_429.jpg | crack | mbdd-crack-004 | Facade crack, UAV survey crop (multi-material building walls) | |
| 13 | data/raw/codebrim/background/background__image_0000489_crop_0000002.png | no_defect | sdnet-cracked-001 | Cracked concrete surface patch | |
| 14 | data/raw/insulator/pollution_flashover/pollution_flashover__train_160516_4.jpg | insulator_damage | insu-pollution-flashover-001 | Pollution-flashover damage on a grid transmission insulator | |
| 15 | data/raw/insulator/pollution_flashover/pollution_flashover__train_17035199h_1.jpg | insulator_damage | insu-pollution-flashover-003 | Pollution-flashover damage on a grid transmission insulator | |
| 16 | data/raw/sdnet2018/cracked/W__CW__7132-227.jpg | crack | sdnet-cracked-001 | Cracked concrete surface patch | |
| 17 | data/raw/bd3/minor_crack/BD3_original_dataset__train__minor_crack__cls02_232.jpg | crack | sdnet-cracked-002 | Cracked concrete surface patch | |
| 18 | data/raw/sdnet2018/non_cracked/W__UW__7075-31.jpg | no_defect | sdnet-cracked-001 | Cracked concrete surface patch | |
| 19 | data/raw/insulator/normal/normal__train_170708v_2.jpg | no_defect | mbdd-crack-004 | Facade crack, UAV survey crop (multi-material building walls) | |
| 20 | data/raw/sdnet2018/non_cracked/P__UP__057-152.jpg | no_defect | sdnet-cracked-001 | Cracked concrete surface patch | |
