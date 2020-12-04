subroutine setprob

    implicit none
    character*25 :: fname
    integer :: iunit
    real(kind=8) :: u,beta,mu1,mu2

    common /cparam/ mu1, mu2
 
    ! Set the material parameters for the acoustic equations
    ! Passed to the Riemann solver rp1.f in a common block
 
    iunit = 7
    fname = 'setprob.data'
    ! open the unit with new routine from Clawpack 4.4 to skip over
    ! comment lines starting with #:
    call opendatafile(iunit, fname)


    ! beta for initial conditions:
    read(7,*) mu1
    read(7,*) mu2

end subroutine setprob
