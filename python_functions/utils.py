#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

def ensure_byteorder(var):
    curorder = var.dtype.byteorder
    if curorder == "=":
        return var
    elif curorder == ">" and sys.byteorder == "little":
        return var.newbyteorder().byteswap(inplace=True)
    elif curorder == "<" and sys.byteorder == "big":
        return var.newbyteorder().byteswap(inplace=True)
    return var

