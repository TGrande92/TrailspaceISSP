#!/bin/bash
fgfs \
    --generic=socket,out,10,https://ec2-3-144-130-20.us-east-2.compute.amazonaws.com,6789,udp,FGtoPX4_2 \
    --aircraft=Rascal110-JSBSim \
    --altitude=400 \
    --telnet=5401 \
    --timeofday=noon
