/*  LICENSE
	Copyright (c) 2021, nabla All rights reserved.
	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/
#ifndef NN_OPTIMIZER_H
#define NN_OPTIMIZER_H



/**********************************************************************************************************************************************

	SGD -- Momentum -- NAG(1983) -- AdaGrad(2011) -- RMSprop(2012) ----- AdaDelta(2012)
																	|--- Adam(2014) ---------------- AdaMax(2015)
																	|--- RMSpropGraves(2014)    |--- NAdam(2016)
																	|--- SMORMS3(2015)          |--- Eve(2016)
																								|--- Santa(2016)
																								|--- AMSGrad(2018) ------ AMSBound(2019)
																								|--- AdaBound(2019)
																								|--- AdaBelief(2020)

**********************************************************************************************************************************************/


/*
* optimizerのクラスは実装する際、
* コンストラクタの引数に、<vector3d>weights, <vector2d>bias, <vector3d>dW, <vector2d>dbを参照型で受け取る。 
* メンバ関数 _Init_() をもつ。クラス固有のパラメタの初期化を行う。
* メンバ関数 update() をもつ。重みとバイアスの更新を行う。
*/




#include "optimizer/adabelief.h"
#include "optimizer/adabound.h"
#include "optimizer/adadelta.h"
#include "optimizer/adagrad.h"
#include "optimizer/adam.h"
#include "optimizer/adamax.h"
#include "optimizer/amsbound.h"
#include "optimizer/amsgrad.h"
#include "optimizer/momentum.h"
#include "optimizer/nadam.h"
#include "optimizer/nag.h"
#include "optimizer/rmsprop.h"
#include "optimizer/rmsptopgraves.h"
#include "optimizer/sgd.h"
#include "optimizer/smorms3.h"






#endif //NN_OPTIMIZER_H