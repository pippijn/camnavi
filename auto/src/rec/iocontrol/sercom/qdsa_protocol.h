//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _QDSA_PROTOCOL_H_
#define _QDSA_PROTOCOL_H_

#define SHUTDOWN_VOLTAGE 764 //21.0V
#define SLEEP_VOLTAGE 800 //22.0V

#define SCALE_VOLTAGE24 36.10f
#define SCALE_VOLTAGE12 71.02f
#define SCALE_CURRENT   61.44f

#define FIRMWAREVERSIONMAPPGING0  411 //mapping between the FIRMWAREVERSION bits from the master and the SVN revision
#define FIRMWAREVERSIONMAPPGING1  670
#define FIRMWAREVERSIONMAPPGING2  1000 //version 0 of the EA09 firmware
#define FIRMWAREVERSIONMAPPGING3  0
#define FIRMWAREVERSIONMAPPGING4  0
#define FIRMWAREVERSIONMAPPGING5  0
#define FIRMWAREVERSIONMAPPGING6  0
#define FIRMWAREVERSIONMAPPGING7  0
#define FIRMWAREVERSIONMAPPGING8  0
#define FIRMWAREVERSIONMAPPGING9  0
#define FIRMWAREVERSIONMAPPGING10 0
#define FIRMWAREVERSIONMAPPGING11 0
#define FIRMWAREVERSIONMAPPGING12 0
#define FIRMWAREVERSIONMAPPGING13 0
#define FIRMWAREVERSIONMAPPGING14 0
#define FIRMWAREVERSIONMAPPGING15 0

/*
Abkürzungen:
NB  : Number of Bytes
P2Q : PC zu qDSA
Q2P : qDSA zu PC
P2M : PC zu Master
M2P : Master zu PC
P2S : PC zu Slave
S2P : Slave zu PC
M   : Master
S   : Slave
DO  : digital Output
DI  : digital Input
DV  : desired Velocity
DP  : desired Position
AV  : actual Velocity
AP  : actual Position
KP  : proportional Constant
KI  : integral Constant
KD  : differential Constant
DIR : direction
AD  : ADC
H   : high (bytes 2-9)
L   : low (bytes 0-1)
*/

#define NB_START 3
#define NB_STOP 3
#define START0 'R'
#define START1 'E'
#define START2 'C'
#define STOP0  'r'
#define STOP1  'e'
#define STOP2  'c'
#define NUM_BYTES_PER_PACKET 15
#define ACK_CHAR '#'
#define RESTART_CHAR '?'

#define NUM_SLAVES 4
#define NUM_MASTER_ADC 3
#define NUM_SLAVE_ADC 8

//Beschreibung des Bytestroms PC zu qDSA
#define NUM_BYTES_P2M 1
#define NUM_BYTES_P2S 10
#define NUM_BYTES_P2Q (NUM_BYTES_P2M+NUM_SLAVES*NUM_BYTES_P2S) //41 bytes
#define NB_P2Q_FULL (NB_START + NUM_BYTES_P2Q + NB_STOP)	   //47 bytes
#define P2M_STOP0_POS (NB_START + NUM_BYTES_P2Q)
#define P2M_STOP1_POS (NB_START + NUM_BYTES_P2Q + 1)
#define P2M_STOP2_POS (NB_START + NUM_BYTES_P2Q + 2)

#define M_DO 0
#define EMERGENCY_STOP  0 //0:STOP 1:OK
#define POWER           1 //0:on 1:off
#define M_LED1          2 //0:off 1:on

#define S_DO 0
#define S_BRAKE   0 //0: brake on, 1: brake off
#define S_DO0     1
#define S_DO1     2
#define S_DO2     3
#define S_DO3     4
#define S_R0      5
#define S_LED1    6 //0:off 1:on

#define S_MISCOUT   1
#define MODE        0  //0=Vel, 1=Pos
#define DV_DIR      1  //0=negative 1=positive
#define RESET_TIME  2  //0=negative 1=positive
#define RESET_POS   3 //1=Position auf 0 setzen

#define S_DV   2
#define S_DP_0 3 //bits 0-7
#define S_DP_1 4 //bits 8-15
#define S_DP_2 5 //bits 16-23
#define S_DP_3 6 //bits 24-31

#define S_KP 7
#define KP_DEFAULT 255
#define S_KI 8
#define KI_DEFAULT 255
#define S_KD 9
#define KD_DEFAULT 255

//Beschreibung des Bytestroms qDSA zu PC
#define NUM_BYTES_M2P 11
#define NUM_BYTES_S2P 21
#define NUM_BYTES_Q2P (NUM_BYTES_M2P+NUM_SLAVES*NUM_BYTES_S2P)  //95 bytes
#define NB_Q2P_FULL (NB_START + NUM_BYTES_Q2P + NB_STOP)		//101 bytes
#define Q2P_STOP0_POS (NB_START + NUM_BYTES_Q2P)
#define Q2P_STOP1_POS (NB_START + NUM_BYTES_Q2P + 1)
#define Q2P_STOP2_POS (NB_START + NUM_BYTES_Q2P + 2)

#define M_AD0_H 0 //Strommessung des Gesamtsystems (SCALE_CURRENT)
#define M_AD1_H 1 //Spannungsmessung Batterie 1 (SCALE_VOLTAGE12)
#define M_AD2_H 2 //Spannungsmessung Batterie 1 + 2 (SCALE_VOLTAGE24)
#define M_AD3_H 3
#define M_AD4_H 4
#define M_AD5_H 5
#define M_AD6_H 6
#define M_AD7_H 7
#define M_AD_L0 8 // 0-1:AD0_L 2-3:AD1_L 4-5:AD2_L 6-7:AD3_L
#define M_AD_L1 9 // 0-1:AD4_L 2-3:AD5_L 4-5:AD6_L 6-7:AD7_L
#define M_TIME  10

#define S_AD0_H 0
#define S_AD1_H 1
#define S_AD2_H 2
#define S_AD3_H 3
#define S_AD4_H 4
#define S_AD5_H 5
#define S_AD6_H 6
#define S_AD7_H 7
#define S_AD_L0 8 // 0-1:AD0_L 2-3:AD1_L 4-5:AD2_L 6-7:AD3_L
#define S_AD_L1 9 // 0-1:AD4_L 2-3:AD5_L 4-5:AD6_L 6-7:AD7_L

#define S_MISCIN  10
#define AV_DIR  0 //0=negative 1=positive
//#define AP_DIR  1 //0=negative 1=positive
#define SPIERROR0 2
#define SPIERROR1 3
#define FIRMWAREVERSION3 4 //bit3  written by the Master into the data of Slave0
#define FIRMWAREVERSION2 5 //bit2
#define FIRMWAREVERSION1 6 //bit1
#define FIRMWAREVERSION0 7 //bit0

#define S_AV    11
#define S_AP_0  12 //bits 0-7
#define S_AP_1  13 //bits 8-15
#define S_AP_2  14 //bits 16-23
#define S_AP_3  15 //bits 24-31

#define S_DI    16
#define S_DI0   0 
#define S_DI1   1
#define S_DI2   2
#define S_DI3   3
#define S_BUMPER   4

#define S_TIME_0 17 // bits 0-7
#define S_TIME_1 18 // bits 8-15
#define S_TIME_2 19 // bits 26-23
#define S_TIME_3 20 // bits 24-31

#endif
