#include "drivers_sbus.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <errno.h>
#include <unistd.h>


int main(void)
{
  printf("SBUS SDK TEST\n");
  initSbus();
  while (1)
  {
    if (recSbusData()==1){
      break;
    }
    checkSbusTimeOut();
  }

  printf("SBUS SDK TEST END END\n");
  return 0;
}
