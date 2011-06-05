/**
 * This file is part of the OpenVIDIA project at http://openvidia.sf.net
 * Copyright (C) 2004, James Fung
 *
 * OpenVIDIA is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * OpenVIDIA is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with OpenVIDIA; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 **/

#include <openvidia/openvidia32.h>
#include <errno.h>
#include <assert.h>


#include <iostream>
using namespace std;

int Timer::elapsedTime() {
   return ((tv_end.tv_sec*1000000+tv_end.tv_usec)-
                (tv_start.tv_sec*1000000+tv_start.tv_usec));
}

void Timer::start() {
  gettimeofday(&tv_start,NULL);

}

void Timer::end() {
  gettimeofday(&tv_end,NULL);
}
