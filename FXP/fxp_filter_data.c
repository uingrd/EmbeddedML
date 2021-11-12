const int Y_SHIFT = 12;
const int NUM_TAPS = 85;
const int NUM_X = 1000;
const signed char x_int[] = 
{
    47, -7, 76, 19, 3, 51, 46, -6, 10, -67, 37, 27, 21, -79, -29, -23, 
    -13, -19, 12, -81, -38, -53, -24, -11, 16, -38, 3, -73, -16, 29, -13, 16, 
    43, 48, 46, 19, 30, 18, 57, 108, 34, 13, 32, -38, 34, -6, 14, 15, 
    34, 12, 27, -51, -55, -18, -36, -27, -15, -30, -12, 18, -63, -34, -21, -35, 
    7, -99, 46, -51, -3, 18, -25, 20, 32, 31, -11, 87, 33, -8, 51, 33, 
    47, -82, 73, 33, 31, 9, 43, 46, 19, 50, 3, -18, -43, -33, 7, -25, 
    -37, -12, -65, -48, -23, -13, -14, -19, -10, 21, -44, -30, 29, -46, 28, -50, 
    -5, 39, 12, 17, 48, -30, 32, 44, 24, 12, 9, 42, -29, 35, 3, 5, 
    32, 27, 8, 0, 16, 46, 2, -53, -93, 33, -86, 7, -6, -44, -8, -56, 
    12, -39, 4, 17, -14, 71, 16, 30, -5, 13, 62, -3, -42, 18, 2, 42, 
    49, 8, 20, -1, -2, 30, 39, 20, 40, 37, -2, -23, -40, -54, -14, -30, 
    42, -25, -76, 14, -101, -75, -19, -29, 15, -69, -57, -33, -23, -51, 7, -14, 
    -37, -14, 54, 9, 21, 57, 40, -2, 22, 12, 35, 10, 60, -3, 2, 17, 
    71, 30, 16, -22, 24, 53, 28, 10, -2, -50, -16, -10, -41, 30, -17, 0, 
    -19, -89, -31, -17, 26, 3, 4, 15, 5, -9, -21, 31, 44, 117, 43, 53, 
    23, 48, 58, -5, -19, 60, 57, 27, -18, 14, 60, -60, -55, -10, -19, -63, 
    -53, -11, -47, -57, 17, -25, -20, -1, -16, 12, -7, 46, -25, 7, 1, -16, 
    0, -9, 24, 54, 27, 46, 42, 23, 16, 20, -46, 78, 28, 14, -22, 72, 
    -1, 2, 12, 21, -32, 19, -18, 20, -28, -40, -80, -42, -15, -9, -37, 56, 
    -82, -26, 27, 4, -4, -61, 9, -12, 15, 46, 18, 13, 55, 27, 7, -1, 
    64, 62, 53, 33, 0, 12, 80, 20, 34, -9, 6, 4, -8, -16, -24, -14, 
    18, -29, -26, -18, -24, -14, -12, -34, -60, -24, -46, -1, 4, -36, -14, 53, 
    -2, 0, -24, 21, 52, 96, 71, 40, 64, 66, 41, 22, 105, 67, -18, -26, 
    29, -30, -20, -2, -56, -8, -48, -78, 31, -71, -13, -53, 18, -35, 29, -39, 
    -20, -39, 14, -70, -34, -33, 13, -25, 21, 31, -17, 24, 28, -4, 23, 93, 
    25, 2, 42, -12, 20, 8, 50, 59, 45, 19, -31, 35, 6, -35, -41, -30, 
    -7, -17, -89, -61, -56, -35, -22, -21, -34, -33, -4, -16, -25, 6, -33, -5, 
    55, -25, -31, 72, 7, 23, 84, 14, 10, 51, 23, 39, 14, -2, 27, 23, 
    -29, -18, -56, 4, -7, -26, -38, -22, -44, 7, -69, -74, 34, -79, -67, -15, 
    -13, 0, -10, -22, -26, -27, 44, -4, 26, 77, 10, 24, 40, -17, 45, 48, 
    -5, -13, 15, 15, -29, 22, -6, 18, -13, -66, 23, 24, -28, -63, -19, -31, 
    -31, -25, -97, -64, -15, -5, -35, -30, -48, 33, -11, 20, -21, -6, -6, -10, 
    65, 60, -19, 24, 37, 47, 66, 18, 56, 32, 38, -8, -5, 19, 20, 82, 
    60, -15, -13, -41, -39, -50, -21, -65, -44, -23, -51, -59, -64, 7, 24, -48, 
    -33, -61, 2, -8, 25, -21, 44, 34, 37, 31, 17, 46, -29, 31, 41, 20, 
    32, 65, 25, 54, 44, -14, 71, -32, 75, -39, 2, 15, 38, -7, -68, 3, 
    -39, 7, 20, 14, -76, -40, -5, -32, -45, -15, -51, -8, -40, 42, 23, 0, 
    34, -24, 2, 22, -9, -5, 63, 86, 33, 36, 40, 19, 39, 53, 36, -48, 
    26, -6, -14, -41, -35, 19, -61, -5, -15, -1, -81, -37, -53, -54, 24, 13, 
    -40, -10, -5, -19, -22, -26, 6, -22, 38, 8, 15, 15, -3, 54, 21, 53, 
    21, 47, 55, -42, -57, 38, -32, -21, -22, -68, 50, -42, -4, 17, -70, -38, 
    -32, -22, 21, -13, -53, 18, -21, -43, 31, -14, -66, 45, 45, 6, 11, 39, 
    48, -69, -4, 48, 64, 24, -4, 78, 23, -2, 37, 45, 78, -38, 60, 32, 
    3, 22, 8, 2, 20, 41, -23, 8, -3, -47, -55, 22, 50, -33, -42, -88, 
    59, -27, -15, -11, -2, 43, 47, 28, 57, 26, -16, 34, -32, 22, 44, 14, 
    41, 98, -29, 12, 7, 57, 19, 15, -44, 31, -20, -45, 10, -18, -60, -12, 
    -4, -24, -20, -43, -61, 22, -9, -51, -23, 16, -28, -16, 1, -18, -58, -24, 
    44, -17, 30, 44, 60, 21, 42, -17, 30, 22, 7, 21, 22, 11, -65, 82, 
    32, 0, 44, -3, 0, 0, -44, -25, -71, -33, -48, -48, -10, -39, -71, -37, 
    49, -48, -16, -57, 3, 2, 15, 6, 4, 15, 85, -58, 39, -2, 26, -3, 
    72, 50, 40, 56, 21, 45, 82, 11, 40, -33, 47, -6, 13, -41, -91, -59, 
    -36, -59, -9, -30, -53, -41, -37, -22, -53, 62, -29, -19, 19, 36, 11, 3, 
    8, 74, -5, 12, 8, 9, 98, 32, 4, 20, 0, 79, 26, 8, -30, 19, 
    -17, 5, -17, 17, -41, 12, -23, -45, 7, 3, -11, -57, -31, -12, -43, -30, 
    38, -20, -6, -42, -12, -22, -8, 40, 9, -16, -1, 31, 42, 47, 25, 18, 
    33, -44, -21, -50, 17, -4, 17, 2, 17, 16, 27, -34, -11, -78, -15, -41, 
    -57, -62, 35, -53, -51, -4, -54, -76, -3, -27, -24, 20, -30, -28, 2, -8, 
    -1, 57, -40, 30, 4, 37, 45, 47, 48, -5, 27, 69, -4, 29, 51, 11, 
    -58, -65, 36, 59, 0, -30, -35, -57, 45, -38, 13, -12, -34, -91, -27, 36, 
    -50, 10, -70, -5, 33, -13, -7, -6, 20, 59, -9, 26, 40, 80, 21, 84, 
    36, 37, -13, 15, 54, -24, 43, -6, 9, -7, 28, 31, 46, -37, -60, -43, 
    -3, 0, -51, -74, -58, -26, -9, -47, -2, -3, 12, -37, -33, -36, 15, 26, 
    34, -45, 1, 31, 56, 39, 54, 6, 
};

const signed char taps_int[] = 
{
    18, 19, 19, 19, 19, 18, 17, 15, 12, 10, 7, 3, 0, -3, -7, -10, 
    -13, -16, -19, -21, -22, -23, -24, -24, -23, -22, -20, -17, -15, -11, -8, -4, 
    0, 4, 8, 12, 15, 18, 21, 23, 24, 25, 26, 25, 24, 23, 21, 18, 
    15, 12, 8, 4, 0, -4, -8, -11, -15, -17, -20, -22, -23, -24, -24, -23, 
    -22, -21, -19, -16, -13, -10, -7, -3, 0, 3, 7, 10, 12, 15, 17, 18, 
    19, 19, 19, 19, 18, 
};

