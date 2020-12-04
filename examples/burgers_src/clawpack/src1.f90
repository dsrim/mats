subroutine src1(meqn,mbc,mx,xlower,dx,q,maux,aux,t,dt)

   implicit real*8(a-h,o-z)
   dimension q(meqn, 1-mbc:mx+mbc)
   real(kind=8) :: mu1,mu2

   common /cparam/ mu1,mu2

   ! reactive source term

   do i=1,mx+mbc
      xcell = xlower + (i-0.5d0)*dx
      q(1,i) = q(1,i) - dt*mu1*(q(1,i)-1.0d0)*(q(1,i)-mu2)*q(1,i)
   end do
end subroutine src1
