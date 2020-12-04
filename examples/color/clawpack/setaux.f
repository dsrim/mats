c     ============================================
      subroutine setaux(mbc,mx,xlower,dx,maux,aux)
c     ============================================
c     
c     # set auxiliary arrays 
c     # variable coefficient acoustics
c     #  aux(i,1) = impedance Z in i'th cell
c     
c     # Piecewise constant medium with single interface at x=0
c     # Density and sound speed to left and right are set in setprob.f
c
      implicit none

      integer, intent(in) :: mbc, mx, maux
      double precision, intent(in) :: xlower, dx
      double precision, intent(out) :: aux
      dimension aux(maux, 1-mbc:mx+mbc)

      integer i
      double precision xcell, u, mu1, mu2, mu3, mu4, pi

      common /cparam/ mu1,mu2,mu3,mu4
        
      pi = 4.0d0*datan(1.0d0)

      do i=1-mbc,mx+mbc
       xcell = xlower + (i-0.5d0)*dx
       aux(1,i) = 1.5d0 + mu1*dsin(mu2*xcell)
       aux(1,i) = aux(1,i) + 0.1d0*dcos(mu3*xcell)
      enddo

      return
      end
