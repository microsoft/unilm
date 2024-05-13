OCR_SYMBOL = "<ocr>"
BOI_SYMBOL = "<image>"
EOI_SYMBOL = "</image>"
EOC_SYMBOL = "</chunk>"
SOB_SYMBOL = "<bbox>"
EOB_SYMBOL = "</bbox>"
MD_SYMBOL = "<md>"

SPECIAL_SYMBOLS = [OCR_SYMBOL, BOI_SYMBOL, EOI_SYMBOL, EOC_SYMBOL, SOB_SYMBOL, EOB_SYMBOL, MD_SYMBOL]
for i in range(4096):
    cur_spcial_token_x = f"<x_{i}>"
    cur_spcial_token_y = f"<y_{i}>"
    SPECIAL_SYMBOLS.append(cur_spcial_token_x)
    SPECIAL_SYMBOLS.append(cur_spcial_token_y)