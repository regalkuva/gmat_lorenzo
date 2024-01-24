function outpar = mean_to_osculating(inpar)

% costanti
Req=6378.139;
mi=398600.4415;
J2=0.0010826269;
omegaterra=7.2921158553e-5;

a=inpar(1);
e=inpar(2);
i=inpar(3);
OMEGA=inpar(4);
omega=inpar(5);
M=inpar(6);

gamma2=0.5*J2*((Req/a)^2);
eta=sqrt(1-(e^2));
gamma2primo=gamma2/(eta^4);

% calcolo dell'anomalia eccentrica e dell'anomalia vera dall'anomalia media

E=M;
eps=E-(e*sin(E))-M;
cont=0;
while abs(eps)>1e-15
    cont=cont+1;
    der=1-(e*cos(E));
    E=E-eps/der;
    eps=E-(e*sin(E))-M;
    if cont>1000
        disp('non convergo dopo 1000 iterazioni');
        pause;
    end;
end;
if sin(E)>=0
    ni=acos((cos(E)-e)/(1-(e*cos(E))));  
else         
    ni=2*pi-acos((cos(E)-e)/(1-(e*cos(E))));        
end;

asur=(1+e*cos(ni))/(eta^2);

anew=a+a*gamma2*((3*(cos(i)^2)-1)*((asur^3)-1/(eta^3))+3*(1-(cos(i)^2))*(asur^3)*cos(2*omega+2*ni));

deltae1=gamma2primo/8*e*(eta^2)*cos(2*omega)*(1-11*(cos(i)^2)-40*(cos(i)^4)/(1-5*(cos(i)^2)));

deltae=deltae1+0.5*(eta^2)*(gamma2*((3*(cos(i)^2)-1)/(eta^6)*(e*eta+(e/(1+eta))+3*cos(ni)+3*e*(cos(ni)^2)...
    +(e^2)*(cos(ni)^3))+3*(1-(cos(i)^2))/(eta^6)*(e+3*cos(ni)+3*e*(cos(ni)^2)+(e^2)*(cos(ni)^3))*cos(2*omega+...
    2*ni))-gamma2primo*(1-(cos(i)^2))*(3*cos(2*omega+ni)+cos(2*omega+3*ni)));

deltai=-(e*deltae1/(eta^2)/tan(i))+0.5*gamma2primo*cos(i)*sqrt(1-(cos(i)^2))*(3*cos(2*omega+2*ni)+...
    3*e*cos(2*omega+ni)+e*cos(2*omega+3*ni));

somma=M+omega+OMEGA+gamma2primo/8*(eta^3)*(1-11*(cos(i)^2)-40*(cos(i)^4)/(1-5*(cos(i)^2)))-...
    gamma2primo/16*(2+(e^2)-11*(2+3*(e^2))*(cos(i)^2)-40*(2+5*(e^2))*(cos(i)^4)/(1-5*(cos(i)^2))-400*...
    (e^2)*(cos(i)^6)/((1-5*(cos(i)^2))^2))+gamma2primo/4*(-6*(1-5*(cos(i)^2))*(ni-M+e*sin(ni))+...
    (3-5*(cos(i)^2))*(3*sin(2*omega+2*ni)+3*e*sin(2*omega+ni)+e*sin(2*omega+3*ni)))-...
    gamma2primo/8*(e^2)*cos(i)*(11+80*(cos(i)^2)/(1-5*(cos(i)^2))+200*(cos(i)^4)/((1-5*(cos(i)^2))^2))-...
    gamma2primo/2*cos(i)*(6*(ni-M+e*sin(ni))-3*sin(2*omega+2*ni)-3*e*sin(2*omega+ni)-e*sin(2*omega+3*ni));

edeltaM=gamma2primo/8*e*(eta^3)*(1-11*(cos(i)^2)-40*(cos(i)^4)/(1-5*(cos(i)^2)))-...
    gamma2primo/4*(eta^3)*(2*(3*(cos(i)^2)-1)*(((asur*eta)^2)+asur+1)*sin(ni)+3*(1-(cos(i)^2))*...
    ((-((asur*eta)^2)-asur+1)*sin(2*omega+ni)+(((asur*eta)^2)+asur+1/3)*sin(2*omega+3*ni)));

deltaOMEGA=-gamma2primo/8*(e^2)*cos(i)*(11+80*(cos(i)^2)/(1-5*(cos(i)^2))+200*(cos(i)^4)/((1-5*(cos(i)^2))^2))-...
    gamma2primo/2*cos(i)*(6*(ni-M+e*sin(ni))-3*sin(2*omega+2*ni)-3*e*sin(2*omega+ni)-e*sin(2*omega+3*ni));
    

d1=(e+deltae)*sin(M)+(edeltaM)*cos(M);

d2=(e+deltae)*cos(M)-(edeltaM)*sin(M);

Mnew=atan2(d1,d2); 

enew=sqrt((d1^2)+(d2^2));

d3=(sin(0.5*i)+cos(0.5*i)*0.5*deltai)*sin(OMEGA)+sin(0.5*i)*deltaOMEGA*cos(OMEGA);

d4=(sin(0.5*i)+cos(0.5*i)*0.5*deltai)*cos(OMEGA)-sin(0.5*i)*deltaOMEGA*sin(OMEGA);

OMEGAnew=atan2(d3,d4);

inew=2*asin(sqrt((d3^2)+(d4^2)));

omeganew=somma-Mnew-OMEGAnew;

outpar(1)=anew;
outpar(2)=enew;
outpar(3)=inew;
outpar(4)=OMEGAnew;
outpar(5)=omeganew;
outpar(6)=Mnew;









