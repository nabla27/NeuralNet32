/*  LICENSE
	Copyright (c) 2021, nabla All rights reserved.
	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/
#ifndef NEURALNET32_NN_H
#define NEURALNET32_NN_H







#include "io/iotxt.h"
#include "io/ioxml.h"
#ifdef USE_OPENCV
#include "io/ioimg.h"
#endif

#ifndef SPECIFY_OPTIMIZER
#include "nn/optimizer.h"
#endif

#include "nn/activation.h"
#include "nn/layerset.h"
#include "nn/network.h"
#include "nn/trainer.h"

#include "vec/function.h"
#include "vec/operator.h"
#include "vec/preprocessing.h"










#endif