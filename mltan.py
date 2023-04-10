# mltan.py

# Conversion to/from mean local time of the ascending node (MLTAN)
# and right ascension of the ascending node (RAAN).

# Precision solar ephemeris.
# Meeus equation-of-time algorithm.

# Created by David Eagle for MATLAB, November 13, 2012
# https://www.mathworks.com/matlabcentral/fileexchange/39085-mean-local-time-of-the-ascending-node

# Ported to Python by Brendan Curley, March 07, 2023

# Copyright (c) 2023, Brendan Curley
# Copyright (c) 2012, David Eagle
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in
#       the documentation and/or other materials provided with the distribution
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

###############################

from datetime import datetime, timedelta

import numpy as np


def atan3(a, b):
    # four quadrant inverse tangent

    # input
    #  a = sine of angle
    #  b = cosine of angle

    # output
    #  y = angle (radians; 0 =< c <= 2 * pi)

    ###############################

    epsilon = 1e-10

    pidiv2 = 0.5 * np.pi

    if np.abs(a) < epsilon:
        y = (1 - np.sign(b)) * pidiv2
        return y
    else:
        c = (2 - np.sign(a)) * pidiv2

    if np.abs(b) < epsilon:
        y = c
        return y
    else:
        y = c + np.sign(a) * np.sign(b) * (np.abs(np.arctan(a / b)) - pidiv2)

    return y


def funarg(t):
    # this function computes fundamental arguments (mean elements)
    # of the sun and moon.  see seidelmann (1982) celestial
    # mechanics 27, 79-106 (1980 iau theory of nutation).

    # input
    #      t      = tdb time in julian centuries since j2000.0 (in)

    # output
    #      el     = mean anomaly of the moon in radians
    #               at date tjd (out)
    #      elprim = mean anomaly of the sun in radians
    #               at date tjd (out)
    #      f      = mean longitude of the moon minus mean longitude
    #               of the moon's ascending node in radians
    #               at date tjd (out)
    #      d      = mean elongation of the moon from the sun in
    #               radians at date tjd (out)
    #      omega  = mean longitude of the moon's ascending node
    #               in radians at date tjd (out)

    ###############################

    seccon = 206264.8062470964

    rev = 1296000

    # compute fundamental arguments in arcseconds
    arg = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    arg[0] = ((+ 0.064 * t + 31.31) * t + 715922.633) * t + 485866.733 + np.mod(1325.0 * t, 1.0) * rev
    arg[0] = np.mod(arg[0], rev)

    arg[1] = ((- 0.012 * t - 0.577) * t + 1292581.224) * t + 1287099.804 + np.mod(99.0 * t, 1.0) * rev
    arg[1] = np.mod(arg[1], rev)

    arg[2] = ((+ 0.011 * t - 13.257) * t + 295263.137) * t + 335778.877 + np.mod(1342.0 * t, 1.0) * rev
    arg[2] = np.mod(arg[2], rev)

    arg[3] = ((+ 0.019 * t - 6.891) * t + 1105601.328) * t + 1072261.307 + np.mod(1236.0 * t, 1.0) * rev
    arg[3] = np.mod(arg[3], rev)

    arg[4] = ((0.008 * t + 7.455) * t - 482890.539) * t + 450160.28 - np.mod(5.0 * t, 1.0) * rev
    arg[4] = np.mod(arg[4], rev)

    # convert arguments to radians
    for i in range(5):
        arg[i] = np.mod(arg[i], rev)
        if arg[i] < 0.0:
            arg[i] = arg[i] + rev
        arg[i] = arg[i] / seccon

    el = arg[0]
    elprim = arg[1]
    f = arg[2]
    d = arg[3]
    omega = arg[4]

    return el, elprim, f, d, omega


def gast2(tjdh, tjdl, k):
    # this function computes the greenwich sidereal time
    # (either mean or apparent) at julian date tjdh + tjdl

    # nutation parameters from function nod

    # input
    #  tjdh = julian date, high-order part
    #  tjdl = julian date, low-order part
    #         julian date may be split at any point, but for
    #         highest precision, set tjdh to be the integral part of
    #         the julian date, and set tjdl to be the fractional part
    #  k    = time selection code
    #         set k=0 for greenwich mean sidereal time
    #         set k=1 for greenwich apparent sidereal time

    # output
    #  gst = greenwich (mean or apparent) sidereal time in hours

    ###############################

    seccon = 206264.8062470964

    t0 = 2451545.0

    tjd = tjdh + tjdl
    th = (tjdh - t0) / 36525.0
    tl = tjdl / 36525.0
    t = th + tl
    t2 = t * t
    t3 = t2 * t

    # for apparent sidereal time, obtain equation of the equinoxes
    eqeq = 0.0

    if k == 1:
        # obtain nutation parameters in seconds of arc
        psi, eps = nod(tjd)
        # compute mean obliquity of the ecliptic in seconds of arc
        obm = 84381.448 - 46.815 * t - 0.00059 * t2 + 0.001813 * t3
        # compute true obliquity of the ecliptic in seconds of arc
        obt = obm + eps
        # compute equation of the equinoxes in seconds of time
        eqeq = psi / 15.0 * np.cos(obt / seccon)

    st = eqeq - 6.2e-06 * t3 + 0.093104 * t2 + 67310.54841 + 8640184.812866 * tl + 3155760000 * tl + 8640184.812866 \
         * th + 3155760000 * th

    gst = np.mod(st / 3600.0, 24.0)

    if gst < 0.0:
        gst = gst + 24

    return gst


def gdate(jdate):
    # convert Julian date to Gregorian (calendar) date

    # input
    #  jdate = julian day

    # output
    #  month = calendar month [1 - 12]
    #  day   = calendar day [1 - 31]
    #  year  = calendar year [yyyy]

    #  note: day may include fractional part

    ###############################

    jd = jdate

    z = np.fix(jd + 0.5)
    fday = jd + 0.5 - z

    if fday < 0:
        fday = fday + 1
        z = z - 1

    if z < 2299161:
        a = z
    else:
        alpha = np.floor((z - 1867216.25) / 36524.25)
        a = z + 1 + alpha - np.floor(alpha / 4)

    b = a + 1524
    c = np.fix((b - 122.1) / 365.25)
    d = np.fix(365.25 * c)
    e = np.fix((b - d) / 30.6001)
    day = b - d - np.fix(30.6001 * e) + fday

    if e < 14:
        month = e - 1
    else:
        month = e - 13

    if month > 2:
        year = c - 4716
    else:
        year = c - 4715

    return month, day, year


def getdate():
    # interactive request and input of calendar date

    # output
    #  m = calendar month
    #  d = calendar day
    #  y = calendar year

    ###############################

    for itry in range(5):
        print('\nplease input the calendar date')
        print('\n(1 <= month <= 12, 1 <= day <= 31, year = all digits!)\n')
        cdstr = input('? ')
        mdy = cdstr.split(',')
        # extract month, day and year
        m = float(mdy[0])
        d = float(mdy[1])
        y = float(mdy[2])
        if 1 <= m <= 12 and 1 <= d <= 31:
            break

    return m, d, y


def getmltan():
    # interactive request and input of mltan

    # output
    #  uthr  = universal time (hours)
    #  utmin = universal time (minutes)
    #  utsec = universal time (seconds)

    ###############################

    for itry in range(5):
        print('\nplease input the mean local time of the ascending node crossing')
        print('\n(0 <= hours <= 24, 0 <= minutes <= 60, 0 <= seconds <= 60)\n')
        utstr = input('? ')
        hms = utstr.split(',')
        # extract hours, minutes and seconds
        uthr = float(hms[0])
        utmin = float(hms[1])
        utsec = float(hms[2])
        # check for valid inputs
        if 0 <= uthr <= 24 and 0 <= utmin <= 60 and 0 <= utsec <= 60:
            break

    return uthr, utmin, utsec


def gettime():
    # interactive request and input of universal time

    # output
    #  uthr  = universal time (hours)
    #  utmin = universal time (minutes)
    #  utsec = universal time (seconds)

    ###############################

    for itry in range(5):
        print('\nplease input the universal time')
        print('\n(0 <= hours <= 24, 0 <= minutes <= 60, 0 <= seconds <= 60)\n')
        utstr = input('? ')
        hms = utstr.split(',')
        # extract hours, minutes and seconds
        uthr = float(hms[0])
        utmin = float(hms[1])
        utsec = float(hms[2])
        # check for valid inputs
        if 0 <= uthr <= 24 and 0 <= utmin <= 60 and 0 <= utsec <= 60:
            break

    return uthr, utmin, utsec


def jd2str(jdate):
    # convert Julian date to string equivalent
    # calendar date and universal time

    # input
    #  jdate = Julian date

    # output
    #  cdstr = calendar date string
    #  utstr = universal time string

    ###############################

    month, day, year = gdate(jdate)

    # datetime object
    sdn = datetime(int(year), int(month), int(np.fix(day))) + timedelta(day - np.fix(day))

    # create calendar date string
    cdstr = sdn.strftime('%d-%b-%Y')

    # create universal time string
    utstr = sdn.strftime('%H:%M:%S.%f')

    return cdstr, utstr


def julian(month, day, year):
    # Julian date

    # Input
    #  month = calendar month [1 - 12]
    #  day   = calendar day [1 - 31]
    #  year  = calendar year [yyyy]

    # Output
    #  jdate = Julian date

    # special notes
    #  (1) calendar year must include all digits
    #  (2) will report October 5, 1582 to October 14, 1582
    #      as invalid calendar dates and stop

    ###################################

    y = year
    m = month
    b = 0
    c = 0

    if m <= 2:
        y = y - 1
        m = m + 12

    if y < 0:
        c = - 0.75

    # check for valid calendar date
    if year < 1582:
        pass
    elif year > 1582:
        a = np.fix(y / 100)
        b = 2 - a + np.floor(a / 4)
    elif month < 10:
        pass
    elif month > 10:
        a = np.fix(y / 100)
        b = 2 - a + np.floor(a / 4)
    elif day <= 4:
        pass
    elif day > 14:
        a = np.fix(y / 100)
        b = 2 - a + np.floor(a / 4)
    else:
        print('\n\n  this is an invalid calendar date!!\n')
        exit()

    jd = np.fix(365.25 * y + c) + np.fix(30.6001 * (m + 1))

    jdate = jd + day + b + 1720994.5

    return jdate


def mltan2raan(jdate, mltan):
    # convert mean local time of the ascending node (MLTAN)
    # to right ascension of the ascending node (RAAN)

    # input
    #  jdate = UTC Julian date of ascending node crossing
    #  mltan = local time of ascending node (hours)

    # output
    #  raan = right ascension of the ascending node (radians)
    #         (0 <= raan <= 2 pi)
    #  rasc_ms = right ascension of the mean sun (radians)
    #  eot = equation of time (radians)

    ###############################

    pi2 = 2.0 * np.pi

    # conversion factors
    dtr = np.pi / 180.0
    atr = dtr / 3600.0

    # compute apparent right ascension of the sun (radians)
    rasc_ts, decl, rsun = sun2(jdate)

    ############################
    # equation of time (radians)
    ############################

    # mean longitude of the sun (radians)
    t = (jdate - 2451545) / 365250
    t2 = t * t
    t3 = t * t * t
    t4 = t * t * t * t
    t5 = t * t * t * t * t
    l0 = dtr * np.mod(280.4664567 + 360007.6982779 * t + 0.03032028 * t2 + t3 / 49931.0 - t4 / 15299.0 - t5 / 1988000.0,
                      360.0)
    # nutations
    psi, eps = nod(jdate)

    # compute mean obliquity of the ecliptic in radians
    t = (jdate - 2451545.0) / 36525.0
    t2 = t * t
    t3 = t2 * t
    obm = atr * (84381.448 - 46.815 * t - 0.00059 * t2 + 0.001813 * t3)

    # compute true obliquity of the ecliptic in radians
    obt = obm + atr * eps
    eot = l0 - dtr * 0.0057183 - rasc_ts + atr * psi * np.cos(obt)

    # right ascension of the mean sun (radians)
    rasc_ms = np.mod(rasc_ts + eot, pi2)

    # right ascension of the ascending node
    # based on local mean solar time (radians)
    raan = np.mod(rasc_ms + dtr * 15.0 * (mltan - 12.0), pi2)

    return raan, rasc_ms, eot


def nod(jdate):
    # this function evaluates the nutation series and returns the
    # values for nutation in longitude and nutation in obliquity.
    # wahr nutation series for axis b for gilbert & dziewonski earth
    # model 1066a. see seidelmann (1982) celestial mechanics 27,
    # 79-106. 1980 iau theory of nutation.

    # jdate = tdb julian date (in)
    # dpsi  = nutation in longitude in arcseconds (out)
    # deps  = nutation in obliquity in arcseconds (out)

    ###############################

    # coefficients
    xnod0 = np.array(
        [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, -1, 1, 0, -1, -1, 1, 2, -2, 0, 2, 2, 1, 0, 0, -1, 0, 0, -1, 0, 1, 0, 2, -1,
         1, 0, 0, 1, 0, -2, 0, 2, 1, 1, 0, 0, 2, 1, 1, 0, 0, 1, 2, 0, 1, 1, -1, 0, 1, 3, -2, 1, -1, 1, -1, 0, -2, 2, 3,
         1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 2, 0, 0, -2, 2, 0, 0, 0, 0, 1, 3, -2, -1, 0, 0, -1, 2, 2, 2, 2, 1, -1, -1,
         0])
    xnod1 = np.array(
        [0, 0, 0, 0, 1, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 1, 0, -1, 0, 0, 0,
         -1, 0, 1, 1, 0, 0, 0, 0, 0, 0, -1, 0, -1, 0, 0, 1, 0, 0, 1, 1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0,
         1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, -1, 0, -1, 1])
    xnod2 = np.array(
        [0, 2, 2, 0, 0, 0, 2, 2, 2, 2, 0, 2, 2, 0, 0, 2, 0, 2, 0, 2, 2, 2, 0, 2, 2, 2, 2, 0, 2, 0, 0, 0, 0, -2, 2, 2, 2,
         2, 0, 2, 0, 0, 2, 0, 2, 0, 2, 2, 0, 0, 0, 0, -2, 0, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 0, 0, 0, 2, 2, 0, 2,
         0, 0, 2, -2, -2, -2, 2, 0, 0, 2, 2, 2, 2, 2, -2, 4, 0, 2, 2, 2, 0, -2, 2, 4, 0, 0, 2, -2, 0, 0, 0, 0])
    xnod3 = np.array(
        [0, -2, 0, 0, 0, 0, -2, 0, 0, -2, -2, -2, 0, 0, 2, 2, 0, 0, -2, 0, 2, 0, 0, -2, 0, -2, 0, 0, -2, 2, 0, -2, 0, 0,
         2, 2, 0, 2, -2, 0, 2, 2, -2, 2, -2, -2, -2, 0, 0, -1, 1, -2, 0, -2, -2, 0, -1, 2, 2, 0, 0, 0, 0, 4, 0, -2, -2,
         0, 0, 0, 0, 1, 2, 2, -2, 2, -2, 2, 2, -2, -2, -4, -4, 4, -1, 4, 2, 0, 0, -2, 0, -2, -2, 2, 0, 2, 0, 0, -2, 2,
         -2, 0, -2, 1, 2, 1])
    xnod4 = np.array(
        [1, 2, 2, 2, 0, 0, 2, 1, 2, 2, 0, 1, 2, 1, 0, 2, 1, 1, 0, 1, 2, 2, 0, 2, 0, 0, 1, 0, 2, 1, 1, 1, 1, 0, 1, 2, 2,
         1, 0, 2, 1, 1, 2, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 2, 2, 2, 2, 2, 0, 2, 2, 1, 1, 1, 1, 0, 2, 2, 1, 1,
         1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 0, 0, 1, 1, 0, 1, 1, 0])
    xnod5 = np.array(
        [-171996, -13187, -2274, 2062, 1426, 712, -517, -386, -301, 217, -158, 129, 123, 63, 63, -59, -58, -51, 48, 46,
         -38, -31, 29, 29, 26, -22, 21, 17, -16, 16, -15, -13, -12, 11, -10, -8, -7, -7, -7, 7, -6, -6, 6, 6, 6, -5, -5,
         -5, 5, -4, -4, -4, 4, 4, 4, -3, -3, -3, -3, -3, -3, -3, 3, -2, -2, -2, -2, -2, 2, 2, 2, 2, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    xnod6 = np.array(
        [-174.2, -1.6, -0.2, 0.2, -3.4, 0.1, 1.2, -0.4, 0, -0.5, 0, 0.1, 0, 0.1, 0, 0, -0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, -0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0])
    xnod7 = np.array(
        [92025, 5736, 977, -895, 54, -7, 224, 200, 129, -95, -1, -70, -53, -33, -2, 26, 32, 27, 1, -24, 16, 13, -1, -12,
         -1, 0, -10, 0, 7, -8, 9, 7, 6, 0, 5, 3, 3, 3, 0, -3, 3, 3, -3, 0, -3, 3, 3, 3, 0, 0, 0, 0, 0, -2, -2, 0, 0, 1,
         1, 1, 1, 1, 0, 1, 1, 1, 1, 1, -1, 0, -1, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0,
         -1, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0])
    xnod8 = np.array(
        [8.9, -3.1, -0.5, 0.5, -0.1, 0, -0.6, 0, -0.1, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0])

    # time argument in julian centuries
    tjcent = (jdate - 2451545.0) / 36525.0

    # get fundamental arguments
    l, lp, f, d, om = funarg(tjcent)

    # sum nutation series terms, from smallest to largest
    dpsi = 0.0
    deps = 0.0
    for j in range(1, 107):
        i = 106 - j
        # formation of multiples of arguments
        arg = xnod0[i] * l + xnod1[i] * lp + xnod2[i] * f + xnod3[i] * d + xnod4[i] * om
        # evaluate nutation
        dpsi = (xnod5[i] + xnod6[i] * tjcent) * np.sin(arg) + dpsi
        deps = (xnod7[i] + xnod8[i] * tjcent) * np.cos(arg) + deps

    dpsi = dpsi * 1e-4
    deps = deps * 1e-4

    return dpsi, deps


def raan2mltan(jdate, raan):
    # convert right ascension of the ascending node (RAAN)
    # to local time of the ascending node (LTAN)

    # input
    #  jdate = UTC Julian date of ascending node crossing
    #  raan  = right ascension of the ascending node (radians)

    # output
    #  mltan = mean local time of the ascending node (hours)
    #  rasc_ms = right ascension of the mean sun (radians)
    #  eot = equation of time (radians)

    ###############################

    pi2 = 2.0 * np.pi

    # conversion factors
    dtr = np.pi / 180.0
    rtd = 180.0 / np.pi
    atr = dtr / 3600.0

    # compute apparent right ascension of the sun (radians)
    rasc_ts, decl, rsun = sun2(jdate)

    ############################
    # equation of time (radians)
    ############################

    # mean longitude of the sun (radians)
    t = (jdate - 2451545) / 365250
    t2 = t * t
    t3 = t * t * t
    t4 = t * t * t * t
    t5 = t * t * t * t * t
    l0 = dtr * np.mod(280.4664567 + 360007.6982779 * t + 0.03032028 * t2 + t3 / 49931.0 - t4 / 15299.0 - t5 / 1988000.0,
                      360.0)
    # nutations
    psi, eps = nod(jdate)

    # compute mean obliquity of the ecliptic in radians
    t = (jdate - 2451545.0) / 36525.0
    t2 = t * t
    t3 = t2 * t
    obm = atr * (84381.448 - 46.815 * t - 0.00059 * t2 + 0.001813 * t3)

    # compute true obliquity of the ecliptic in radians
    obt = obm + atr * eps
    eot = l0 - dtr * 0.0057183 - rasc_ts + atr * psi * np.cos(obt)

    # right ascension of the mean sun (radians)
    rasc_ms = np.mod(rasc_ts + eot, pi2)

    # mean local time of the ascending node (hours)
    mltan = rtd * (raan - rasc_ms) / 15.0 + 12.0

    return mltan, rasc_ms, eot


def sun2(jdate):
    # precision ephemeris of the Sun

    # input
    #  jdate = julian ephemeris date

    # output
    #  rasc = right ascension of the Sun (radians)
    #         (0 <= rasc <= 2 pi)
    #  decl = declination of the Sun (radians)
    #         (-pi/2 <= decl <= pi/2)
    #  rsun = eci position vector of the Sun (km)

    # note
    #  coordinates are inertial, geocentric,
    #  equatorial and true-of-date

    ###############################

    # coefficients
    sl = np.array(
        [403406, 195207, 119433, 112392, 3891, 2819, 1721, 0, 660, 350, 334, 314, 268, 242, 234, 158, 132, 129,
         114, 99, 93, 86, 78, 72, 68, 64, 46, 38, 37, 32, 29, 28, 27, 27, 25, 24, 21, 21, 20, 18, 17, 14, 13, 13,
         13, 12, 10, 10, 10, 10])
    sr = np.array(
        [0, -97597, -59715, -56188, -1556, -1126, -861, 941, -264, -163, 0, 309, -158, 0, -54, 0, -93, -20, 0,
         -47, 0, 0, -33, -32, 0, -10, -16, 0, 0, -24, -13, 0, -9, 0, -17, -11, 0, 31, -10, 0, -12, 0, -5, 0,
         0, 0, 0, 0, 0, -9])
    sa = np.array(
        [4.721964, 5.937458, 1.115589, 5.781616, 5.5474, 1.512, 4.1897, 1.163, 5.415, 4.315, 4.553, 5.198, 5.989,
         2.911, 1.423, 0.061, 2.317, 3.193, 2.828, 0.52, 4.65, 4.35, 2.75, 4.5, 3.23, 1.22, 0.14, 3.44, 4.37,
         1.14, 2.84, 5.96, 5.09, 1.72, 2.56, 1.92, 0.09, 5.98, 4.03, 4.27, 0.79, 4.24, 2.01, 2.65, 4.98, 0.93,
         2.21, 3.59, 1.5, 2.55])
    sb = np.array(
        [1.621043, 62830.34807, 62830.82152, 62829.6343, 125660.5691, 125660.9845, 62832.4766, 0.813, 125659.31,
         57533.85, -33.931, 777137.715, 78604.191, 5.412, 39302.098, -34.861, 115067.698, 15774.337, 5296.67,
         58849.27, 5296.11, -3980.7, 52237.69, 55076.47, 261.08, 15773.85, 188491.03, -7756.55, 264.89, 117906.27,
         55075.75, -7961.39, 188489.81, 2132.19, 109771.03, 54868.56, 25443.93, -55731.43, 60697.74, 2132.79,
         109771.63, -7752.82, 188491.91, 207.81, 29424.63, -7.99, 46941.14, -68.29, 21463.25, 157208.4])

    # fundamental time argument
    u = (jdate - 2451545) / 3652500

    # compute nutation in longitude
    a1 = 2.18 + u * (- 3375.7 + u * 0.36)
    a2 = 3.51 + u * (125666.39 + u * 0.1)

    psi = 1e-07 * (- 834 * np.sin(a1) - 64 * np.sin(a2))

    # compute nutation in obliquity
    deps = 1e-07 * u * (- 226938 + u * (- 75 + u * (96926 + u * (- 2491 - u * 12104))))

    meps = 1e-07 * (4090928 + 446 * np.cos(a1) + 28 * np.cos(a2))

    eps = meps + deps

    seps = np.sin(eps)
    ceps = np.cos(eps)

    dl = 0
    dr = 0

    for i in range(50):
        w = sa[i] + sb[i] * u
        dl = dl + sl[i] * np.sin(w)
        if sr[i] != 0:
            dr = dr + sr[i] * np.cos(w)

    dl = np.mod(dl * 1e-07 + 4.9353929 + 62833.196168 * u, 2.0 * np.pi)

    dr = 149597870.691 * (dr * 1e-07 + 1.0001026)

    # geocentric ecliptic position vector of the Sun
    rlsun = np.mod(dl + 1e-07 * (- 993 + 17 * np.cos(3.1 + 62830.14 * u)) + psi, 2.0 * np.pi)

    rb = 0

    # compute declination and right ascension
    cl = np.cos(rlsun)
    sl = np.sin(rlsun)
    cb = np.cos(rb)
    sb = np.sin(rb)

    decl = np.arcsin(ceps * sb + seps * cb * sl)

    sra = - seps * sb + ceps * cb * sl
    cra = cb * cl

    rasc = atan3(sra, cra)

    # geocentric equatorial position vector of the Sun
    rsun = np.array([0.0, 0.0, 0.0])
    rsun[0] = dr * np.cos(rasc) * np.cos(decl)
    rsun[1] = dr * np.sin(rasc) * np.cos(decl)
    rsun[2] = dr * np.sin(decl)

    return rasc, decl, rsun


###############################

if __name__ == '__main__':
    # conversion constants
    pi2 = 2.0 * np.pi
    dtr = np.pi / 180.0
    rtd = 180.0 / np.pi
    print('\n\nMLTAN/RAAN relationship')
    print('\n-----------------------')

    # request type of conversion
    while 1:
        print('\nplease select the type of conversion')
        print('\n <1> MLTAN to RAAN')
        print('\n <2> RAAN to MLTAN')
        print('\nselection (1 or 2)')
        choice = int(input('? '))
        if 1 <= choice <= 2:
            break

    ##################################
    # request ascending node UTC epoch
    ##################################

    print('\n\nascending node UTC epoch')
    month, day, year = getdate()
    utc_an_hr, utc_an_min, utc_an_sec = gettime()
    dday = utc_an_hr / 24 + utc_an_min / 1440 + utc_an_sec / 86400.0

    # julian date of ascending node crossing
    jdate_an = julian(month, day + dday, year)
    cdstr_an, utstr_an = jd2str(jdate_an)

    # greenwich apparent sidereal time at ascending node
    gast_an = gast2(jdate_an, 0.0, 1)

    if choice == 1:
        ###############
        # mltan to raan
        ###############
        mltan_hr, mltan_min, mltan_sec = getmltan()
        mltan_an = mltan_hr + mltan_min / 60.0 + mltan_sec / 3600.0
        # compute raan from mltan
        raan, rasc_ms, eot = mltan2raan(jdate_an, mltan_an)
        dday = mltan_an / 24.0
        # working julian date
        jdate_wrk = julian(month, day + dday, year)
        cdstr_wrk, utstr_wrk = jd2str(jdate_wrk)
    else:
        ###############
        # raan to mltan
        ###############
        while 1:
            print('\nplease input the RAAN (degrees)')
            raan = float(input('? '))
            if 0 <= raan <= 360:
                break

        raan = dtr * raan
        mltan_an, rasc_ms, eot = raan2mltan(jdate_an, raan)
        dday = mltan_an / 24.0
        # working julian date
        jdate_wrk = julian(month, day + dday, year)
        cdstr_wrk, utstr_wrk = jd2str(jdate_wrk)

    # east longitude of the ascending node
    elan = np.mod(raan - 2.0 * np.pi * gast_an / 24.0, 2 * np.pi)

    # print results
    if choice == 1:
        print('\n\nMLTAN to RAAN conversion')
    else:
        print('\n\nRAAN to MLTAN conversion')

    print('\nascending node calendar date             ' + cdstr_an)
    print('\nascending node universal time            ' + utstr_an)
    print('\nmean local time of the ascending node    ' + utstr_wrk)
    print('\nright ascension of the ascending node  %12.6f degrees' % (rtd * raan))
    print('\neast longitude of the ascending node   %12.6f degrees' % (rtd * elan))
    print('\nright ascension of the mean sun        %12.6f degrees' % (rtd * rasc_ms))
    print('\nGreenwich apparent sidereal time       %12.6f degrees' % (360.0 * gast_an / 24.0))
    print('\nequation of time                       %12.6f minutes\n' % (4.0 * rtd * eot))
