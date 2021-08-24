/*  LICENSE
	Copyright (c) 2021, nabla All rights reserved.
	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/
#ifndef UTIL_TIMER_H
#define UTIL_TIMER_H

#include <time.h>
#include <iostream>
#include <conio.h>





//�w�肵�����Ԃ̍���(�~���b)���o�ߓ����ɕϊ����A�W���o�͂���
void show_et(const clock_t start, const clock_t end)
{
	const clock_t diff = (end - start) / CLOCKS_PER_SEC;
	const clock_t minutes = diff / 60;
	const clock_t hours = minutes / 60;
	const clock_t days = hours / 24;
	std::cout << "time: "
		<< days << "days "
		<< hours - days * 24 << "hours "
		<< minutes - hours * 60 << "minutes "
		<< diff - minutes * 60 << "seconds "
		<< std::endl;
	(void)_getch();
}



#endif