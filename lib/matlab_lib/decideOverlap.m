function [D,ratio] = decideOverlap(Reframe,GTframe,thred)

x1 = single(Reframe(1));
y1 = single(Reframe(2));
width1 = single(Reframe(3));
height1 = single(Reframe(4));

x2 = single(GTframe(1));
y2 = single(GTframe(2));
width2 = single(GTframe(3));
height2 = single(GTframe(4));

endx = max(x1+width1,x2+width2);
startx = min(x1,x2);
width = width1+width2-(endx-startx);

endy = max(y1+height1,y2+height2);
starty = min(y1,y2);
height = height1+height2-(endy-starty);

if width<=0||height<=0
    D = 0;
    ratio = 0;
else
    Area = width*height;
    Area1 = width1*height1;
    Area2 = width2*height2;
    ratio = single(Area)/single(Area1+Area2-Area);
    if ratio>=thred
        D = 1;
    else
        D = 0;
    end
end

end