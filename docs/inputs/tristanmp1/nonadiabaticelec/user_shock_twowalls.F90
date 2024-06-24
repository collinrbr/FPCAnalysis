!
! User module
!
! This module contains functions that may be altered by a user of the code, and that are called 
! if caseinit variable is set to a number greater than 0. The functions that are going to be 
! called in such a case are: SetEMFieldsUser, ..., ...
!
! If a user wants to alter the functions that are called he/she may also alter the module m_overload
! which branches with the variable caseinit.
!
!By Aaron Tran and Lorenzo Sironi


#ifdef twoD 

module m_user

	use m_globaldata
	use m_system
	use m_aux
	use m_communications
	use m_fields
	use m_particles
	use m_inputparser
	use m_fparser
	use m_domain
	
#else

module m_user_3d

	use m_globaldata_3d
	use m_system_3d
	use m_aux_3d
	use m_communications_3d
	use m_fields_3d
	use m_particles_3d
	use m_inputparser_3d 
	use m_fparser_3d
	use m_domain_3d

#endif




	implicit none
		
	private

!-------------------------------------------------------------------------------
!	PARAMETERS
!-------------------------------------------------------------------------------

	
!-------------------------------------------------------------------------------
!	TYPE DEFINITIONS
!-------------------------------------------------------------------------------

!-------------------------------------------------------------------------------
!	VARIABLES
!-------------------------------------------------------------------------------

	real(sprec) :: temperature_ratio, sigma_ext, bz_ext0, u2z_in, u_sh_in,u_sh_left
        real(sprec) :: injfrac
	character(len=256) :: density_profile
        integer rightclean,rstart_jump,rstep_jump,rstep_first
        integer rightclean_in, slowinj
        integer lstart_jump,lstep_jump,lstep_first

!-------------------------------------------------------------------------------
!	INTERFACE DECLARATIONS
!-------------------------------------------------------------------------------
	
!-------------------------------------------------------------------------------
!	PUBLIC MODIFIERS
!-------------------------------------------------------------------------------

	public :: init_EMfields_user, init_particle_distribution_user, &
	inject_particles_user, read_input_user, field_bc_user, get_external_fields, &
	particle_bc_user

!-------------------------------------------------------------------------------
!	MODULE PROCEDURES AND FUNCTIONS
!-------------------------------------------------------------------------------

	contains



!-------------------------------------------------------------------------------
! 						subroutine read_input_shock		
!									
! Reads any variables related to (or needed by) this module
! 							
!-------------------------------------------------------------------------------

subroutine read_input_user()

	implicit none
	integer lextflds, luserpartbcs
	integer :: lwall

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!CHANGE THIS NAME IF YOU ARE CREATING A NEW USER FILE
!This helps to identify which user file is being compiled through Makefile. 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	if(rank.eq.0)print *, "Using user file user_shock_twowalls.F90"

	call inputpar_getd_def("problem", "temperature_ratio", 1._sprec, Temperature_ratio)

        call inputpar_geti_def("problem", "slowinj", 0, slowinj)
        call inputpar_getd_def("problem", "injfrac", 1._sprec, injfrac)

	call inputpar_gets_def("problem", "density_profile", "0", density_profile)

        call inputpar_geti_def("problem", "rightclean", 1, rightclean_in)
        call inputpar_geti_def("problem", "rstart_jump", 30000, rstart_jump)
        call inputpar_geti_def("problem", "rstep_first", 5000, rstep_first)
        call inputpar_geti_def("problem", "rstep_jump", 5000, rstep_jump)

        call inputpar_geti_def("problem", "leftclean", 0, leftclean)
        call inputpar_geti_def("problem", "lstart_jump", 30000, lstart_jump)
        call inputpar_geti_def("problem", "lstep_first", 5000, lstep_first)
        call inputpar_geti_def("problem", "lstep_jump", 5000, lstep_jump)

	call inputpar_getd_def("problem","sigma_ext",0._sprec,sigma_ext)

	call inputpar_geti_def("problem","external_fields",0,lextflds)

	if(lextflds .eq. 1) then 
	   external_fields =.true.
	else
	   external_fields =.false.
	endif

	if(external_fields) bz_ext0 = sqrt((gamma0-1)*.5*ppc0*c**2*(mi+me)*sigma_ext)

	call inputpar_geti_def("problem","user_part_bcs",0,luserpartbcs)

	if(luserpartbcs .eq. 1) then 
	   user_part_bcs = .true.
	else
	   user_part_bcs = .false.
	endif

	call inputpar_geti_def("problem", "caseinit", 0, caseinit)

	call inputpar_geti_def("problem", "wall", 0, lwall)

	call inputpar_geti_def("problem", "wall", 0, lwall)
	
	call inputpar_getd_def("problem","wallgam",1._sprec,wallgam)

        !read input to set the tangential velocity of the BC
        call inputpar_getd_def("problem", "u2z", 0._sprec, u2z_in)

        !read input to set the injector backspeed
        call inputpar_getd_def("problem", "u_sh", sqrt(gamma0-1), u_sh_in)

        call inputpar_getd_def("problem", "u_sh_left", 0._sprec, u_sh_left) 

	if(wallgam .eq. 0) wallgam=1.	
	if(wallgam<1) wallgam=1./sqrt(1-wallgam**2)


	if (lwall==1) then
		wall=.true.
	else
		wall=.false.
	endif
	
	if(wall) user_part_bcs=.true.

        if (rank .eq. 0) then
           write(*,*) "u2z", u2z_in, "u_sh", u_sh_in, "injfrac", injfrac 
        endif

end subroutine read_input_user

!-------------------------------------------------------------
!     Compute external fields to be added to the mover. 
!     These fields do not evolve via Maxwell Eqs, but can depend on time
!-------------------------------------------------------------
	subroutine get_external_fields(x,y,z,ex_ext, ey_ext, ez_ext, bx_ext,by_ext,bz_ext)
	
	real,intent(inout):: bx_ext,by_ext,bz_ext, ex_ext, ey_ext, ez_ext
	real, intent(in):: x,y,z
	ex_ext=0.
	ey_ext=0.
	ez_ext=0.
	bx_ext=0.
	by_ext=0.
	bz_ext=bz_ext0
	
	end subroutine get_external_fields


!-------------------------------------------------------------------------------
! 						subroutine parse_density_profile_function()
!												
! Parses the mathematical function that defines the density profile, defined in
! the input file as density_profile
!-------------------------------------------------------------------------------

subroutine parse_density_profile_function(use_density_profile)

	implicit none
	
	! dummy variables
	
	logical, intent(out) :: use_density_profile
	
	! local variables
	
	character(len=1), dimension(3) :: vars=(/'x','y','z'/)
	logical, save :: initialized=.false.

	use_density_profile=.true.	
	
	if (density_profile=="0") then
		use_density_profile=.false.
		return
	endif
	
	if (.not. initialized) then	
		call initf(10)
		call parsef (1, density_profile, vars)
		initialized=.true.
	endif
	
end subroutine parse_density_profile_function



!-------------------------------------------------------------------------------
! 						subroutine init_EMfields_shock		 
!												
! Sets the electromagnetic fields of any specific user purpose
!							
!-------------------------------------------------------------------------------

subroutine init_EMfields_user()
	
	! local variables
	
	integer :: i, j, k, jglob, kglob
	real(sprec) :: beta0, betacur
	
	!determine initial magnetic field based on magnetization sigma which 
        !is magnetic energy density/ kinetic energy density
	!this definition works even for nonrelativistic flows. 
	
	btheta=btheta/180.*pi
	bphi=bphi/180.*pi
	
!	Binit=sqrt((gamma0-1 )*ppc0*.5*c**2*(mi+me)*sigma) 
!	if(gamma0 .ge. 2) then 
!		Binit=sqrt((gamma0)*ppc0*.5*c**2*(mi+me)*sigma) !relativistic
!	if(rank .eq. 0) print *, "rank 0: USING RELATIVISTIC BINIT INITIALIZATION"
!	else
		Binit=sqrt((gamma0 -1)*ppc0*.5*c**2*(mi+me)*sigma) !nonrelativistic
	 if(rank .eq. 0) print *, "rank 0: USING NON-RELATIVISTIC BINIT INITIALIZATION"	
!	endif

!	Binit=sqrt( ( wallgam-1 )*ppc0*.5*c**2*(mi+me)*sigma)  !piston frame

        !using full gamma0 to determine initial Binit. This way, B field can be defined even in stationary plasma 
	!For non-rel shocks, it makes more sense to use gamma0-1

	!initialize B field to be set by Binit and the inclination angle -- used for shocks
	do  k=1,mz
		do  j=1,my
			do  i=1,mx

				jglob=j+modulo(rank,sizey)*(myall-5) !global j,k coords in 
				kglob=k+(rank/sizey)*(mzall-5)       !case need global variation of fields
				
				bx(i,j,k)=Binit*cos(btheta) 
				by(i,j,k)=Binit*sin(btheta)*sin(bphi)
				bz(i,j,k)=Binit*sin(btheta)*cos(bphi)

				ex(i,j,k)=0.
				ey(i,j,k)=(-beta)*bz(i,j,k) 
				ez(i,j,k)=-(-beta)*by(i,j,k)

			enddo
		enddo
	enddo
	
end subroutine init_EMfields_user


!-------------------------------------------------------------------------------
! 						subroutine init_particle_distribution_shock()	
!											
! Sets the particle distrubtion for a user defined case
!
!-------------------------------------------------------------------------------

subroutine init_particle_distribution_user()

	implicit none

	! local variables
	
	real(sprec), dimension(pdf_sz) :: func
	real(sprec) :: maxg, delgam1
	integer :: i, n, direction
	real       b00,db0,vpl,vmn,gam, cossq, rad, gamma_drift, delgam_i, delgam_e
	real betap, real, tmp, ppc
	real(dprec) :: Lps,pps 
	real numps,kps,ups, weight
	logical :: use_density_profile
        logical :: existfld
	real(dprec) :: x1,x2,y1,y2,z1,z2

	call parse_density_profile_function(use_density_profile)
	
	call init_split_parts() !set split points for particle splits, not used unless splitpart is set to true in input
	
	pcosthmult=0 !if 0 and 2D run, the Maxwellian distribution corresponding 
                     !to temperature is initialized in 2D, 1 for 3D.  
	             !when sigma > 0, it is set to 3D automatically

	! -------- Set particle boundaries ------------
	!set initial injection points for shocks 
	
	xinject=3
        if (slowinj .eq. 1) then
           if (irestart .eq. 1) then
              inquire(file=frestartfldlap,exist=existfld)
              if (existfld) then !restart file exists, only inject small region
                 xinject2 = min(50., (mx0-2.) )
              else !no restart file, start from scratch, inject large region
                 xinject2 = mx0-2.
              endif
           else ! start from scratch, inject large region
              xinject2= mx0-2.
           endif
        else
           xinject2=min(50.,(mx0-2.))
        endif

        if (rank .eq. 0) print *, "rightclean=", rightclean, " injfrac ",injfrac,  " xinject2=", xinject2

	if(wall) leftwall=15. !reset the location of reflecting wall; need to control it from input file. 
	if(wall) xinject=leftwall+1
	
	! ------------------------------------------

	
	totalpartnum=0 !for purpose of keeping track of the total number of particles injected on this cpu

	gamma_drift=-gamma0 ! negative gamma_drift will send the plasma in the negative direction
	delgam_i=delgam
	delgam_e=delgam*mi/me*Temperature_ratio

	x1=xinject  
	x2=xinject2 

	y1=3. !in global coordinates
	y2=my0-2.  
	z1=3.
	z2=mz0-2. !if 2D, it will reset automatically
	ppc=ppc0 *1  !don't initialize plasma in the box if ppc0=0, only inject in inject_particle_user
	weight=1

	direction=1 !drift along x
	call inject_plasma_region(x1,x2,y1,y2,z1,z2,ppc,&
             gamma_drift,delgam_i,delgam_e,weight,use_density_profile,direction)

	x1in=3 !set the location of planes where particles are removed from simulation, perpendicular to x. 
	x2in=mx0-2 

	call check_overflow()
	call reorder_particles()
	
    
    rightclean = rightclean_in
    if (rank.eq.0 .and. rightclean .eq. 1) then
       write(*,*) "Using right clean!"
    endif
    
end subroutine init_particle_distribution_user


!-------------------------------------------------------------------------------
! 				subroutine inject_particles_shock()					 
!										
! Injects new particles in the simulation on every step. To skip injection, set ppc=0 below
!
!-------------------------------------------------------------------------------

subroutine inject_particles_user()

	implicit none
	real(dprec) :: x1,x2,y1,y2,z1,z2
	real delgam_i, delgam_e, injector_speed, ppc, betainj, gamma_drift, weight
	logical use_density_profile

	integer numrocks,nr,rockinter,direction, ierr
	real(dprec) :: xlen,ylen, xreg
	real ranweight, betadrift
        real backinj, beta0inj, bthetainj, betaguess

	use_density_profile=.false. !no profile possible in injection now

	injectedions=0 !set counter to 0 here, so that all possible other injections on this step add up
	injectedlecs=0

        if (slowinj .eq. 1) then
           betainj = max(beta, .9999)*injfrac
        else
           betainj=max(beta,.99999)	!move injector almost at c always       
        endif
		
	!make moving injection spigot

!	betainj=0 !to stop the injector from moving

!       determine the location of injector
	xinject2=xinject2+c*betainj

	if(xinject2 .gt. mx-2) then !stop expansion of the injector has reached the end of the domain
	   xinject2=mx-2.
	   betainj=0. !injector hit the wall, stop moving the injector
	endif


	injector_speed = betainj
	
	gamma_drift= -gamma0  ! negative gamma_drift will send the plasma in the -x direction
	delgam_i=delgam
	delgam_e=delgam*mi/me*Temperature_ratio
	ppc=ppc0
	weight=1.

	x1=xinject2 
	x2=x1 !x2=x1 if the injection is parallel to x

	y1=3. !global coordinates
	y2=my0-2.   
	z1= 3.
	z2= mz0-2. !if 2D, it will reset automatically

	call inject_from_wall(x1,x2,y1,y2,z1,z2,ppc,gamma_drift,delgam_i, &
	delgam_e,injector_speed,weight,use_density_profile)
	
	x1in=3 !set the location of planes where particles are removed from simulation, perpendicular to x. 
	x2in=mx0-2 !need to set it again because mx0 could be changing between init and inject due to domain enlargement

 ! assuming that I have particles flying at the speed of light along the field
 beta0inj=sqrt(1.-1./gamma0**2)
 bthetainj=atan(tan(btheta)/gamma0)
 betaguess=max((cos(bthetainj)-beta0inj)/(1.-beta0inj*cos(bthetainj)),beta0inj/3.) ! to be set to /2. for 2d unmagnetized simulations
 !backinj=max(betainj-betaguess,0.)!beta !0.33 !1.-betshock
 !old way
 backinj=betainj-u_sh_in
 if(rightclean .eq. 1 .and. lap .ge. rstart_jump) then
    if (lap .eq. rstart_jump) then 
       xinject2=xinject2-backinj*c*rstep_first
    endif
    if (lap .gt. rstart_jump .and. modulo(lap, rstep_jump) .eq. 0) then 
       xinject2=xinject2-backinj*c*rstep_jump
    endif    
    !x2in=xinject2-c*beta
 endif

	
end subroutine inject_particles_user



!-------------------------------------------------------------------------------
! 				subroutine field_bc_shock()	
!										
! Applies boundary conditions specific to user problem. 
! 
!-------------------------------------------------------------------------------

	subroutine field_bc_user()
	implicit none
	integer i,j,k

!reset fields on the right end of the grid, where the plasma is injected
!make it drifting fields, even though there is no plasma there
		
				bz(mx-10:mx,:,:)=binit*sin(btheta)*cos(bphi)
				by(mx-10:mx,:,:)=binit*sin(btheta)*sin(bphi)
				bx(mx-10:mx,:,:)=binit*cos(btheta)
				ey(mx-10:mx,:,:)=(-beta)*bz(mx-10:mx,:,:)
				ez(mx-10:mx,:,:)=(beta)*by(mx-10:mx,:,:)
				ex(mx-10:mx,:,:)=0

    if(rightclean .eq. 1 .and. (lap .eq. rstart_jump+1 .or. &
         (lap .gt. rstart_jump+1 .and. modulo(lap, rstep_jump) .eq. 1))) then
       do  k=1,mz
          do  j=1,my
             do  i=int(xinject2)+1,mx-3
                bx(i,j,k)=Binit*cos(btheta) 
                by(i,j,k)=Binit*sin(btheta)*sin(bphi) 
                bz(i,j,k)=Binit*sin(btheta)*cos(bphi) 
                ex(i,j,k)=0.
                ey(i,j,k)=(-beta)*bz(i,j,k) 
                ez(i,j,k)=-(-beta)*by(i,j,k)            
             enddo
          enddo
       enddo
    endif
    
	!reflecting wall for EM fields

	if(wall) then
		ey(1:10,:,:)=(-u2z_in)*Binit*cos(btheta)*cos(bphi)
		ez(1:10,:,:)=(u2z_in)*Binit*cos(btheta)*sin(bphi)
	endif

	end subroutine field_bc_user

!------------------------------------------------------------------------------

	subroutine particle_bc_user()
	implicit none
	real invgam, gammawall, betawall, gamma, y0, z0, betawall2, gammawall2
	real ycolis,zcolis,q0,tfrac 
	real(dprec) :: x0, xcolis
        real(dprec) :: walloc,walloc0,wall2loc,wall2loc0
	integer n1,i0,i1, iter,wall2counter
	logical in,wall2bool
        real betainj

	!loop over particles to check if they crossed special boundaries, like reflecting walls
	!outflow and periodic conditions are handled automatically in deposit_particles
	!
	!This routine is called after the mover and before deposit, thus allowing to avoid
        ! charge deposition behind walls. 
	
        betainj=max(beta,.99999)	!move injector almost at c always 

        
        wall2bool=.true.
        if(rightclean .eq. 1 .and. (lap .eq. rstart_jump+1 .or. &
             (lap .gt. rstart_jump+1 .and. modulo(lap, rstep_jump) .eq. 1))) then 
           wall2bool=.false.
           x2in=xinject2-c*beta
        endif

	if(wall) then
           wall2counter = 0

	   gammawall=wallgam
	   betawall=sqrt(1.-1/gammawall**2)
	   
	   walloc=leftwall + betawall*c*lap 
	   if(movwin) walloc=walloc-movwinoffset
           walloc=walloc+leftcleanoffset
    
	   do iter=1,2
	      if(iter.eq.1) then 
		 i0=1
		 i1=ions
		 q0=qi
	      else
		 i0=maxhlf+1
		 i1=maxhlf+lecs
		 q0=qe
	      endif
		 
	   do n1=i0,i1

	   if(p(n1)%x .lt. walloc ) then 
	      gamma=sqrt(1+(p(n1)%u**2+p(n1)%v**2+p(n1)%w**2))
	     
	      !this algorithm ignores change in y and z coordinates
	      !during the scattering. Including it can result in rare
	      !conditions where a particle gets stuck in the ghost zones. 
	      !This can be improved. 

	      !unwind x location of particle
	      x0=p(n1)%x-p(n1)%u/gamma*c
	      y0=p(n1)%y !-p(n1)%v/gamma*c
	      z0=p(n1)%z !-p(n1)%w/gamma*c

	      !unwind wall location
	      walloc0=walloc-betawall*c
	      
	      !where did they meet?
	      !tfrac=abs((x0-walloc0)/(betawall*c-p(n1)%u/gamma*c))
	      tfrac=min(abs((x0-walloc0)/max(abs(betawall*c-p(n1)%u/gamma*c),1e-9)),1.)
	      xcolis=x0+p(n1)%u/gamma*c*tfrac
	      ycolis=y0 !+p(n1)%v/gamma*c*tfrac
	      zcolis=z0 !+p(n1)%w/gamma*c*tfrac

	      !deposit current upto intersection
	      q=p(n1)%ch*real(splitratio)**(1.-real(p(n1)%splitlev))*q0
	      call zigzag(xcolis,ycolis,zcolis,x0,y0,z0,in)

	      !reset particle momentum, getting a kick from the wall
	      p(n1)%u=gammawall**2*gamma*(2*betawall - p(n1)%u/gamma*(1 + betawall**2))
	      gamma=sqrt(1+(p(n1)%u**2+p(n1)%v**2+p(n1)%w**2))
	      
	      tfrac=min(abs((p(n1)%x-xcolis)/max(abs(p(n1)%x-x0),1e-9)),1.)
              !move particle from the wall position with the new velocity
 
	      p(n1)%x = xcolis + p(n1)%u/gamma*c * tfrac
	      p(n1)%y = ycolis !+ p(n1)%v/gamma*c * tfrac
	      p(n1)%z = zcolis !+ p(n1)%w/gamma*c * tfrac

	     
!now clean up the piece of trajectory behind the wall, that deposit_particles will be adding when it 
!unwinds the position of the particle by the full timestep. 
	      
	      q=-q
	      call zigzag(xcolis,ycolis,zcolis,p(n1)%x-p(n1)%u/gamma*c, & 
	              p(n1)%y-p(n1)%v/gamma*c,p(n1)%z-p(n1)%w/gamma*c,in)
	      
	      	   endif

             wall2loc = xinject2-beta*c
	      if(wall2bool .and. p(n1)%x .gt. wall2loc+1e6*0.) then
              wall2counter = wall2counter+1

	      gamma=sqrt(1+(p(n1)%u**2+p(n1)%v**2+p(n1)%w**2))
	     
	      !this algorithm ignores change in y and z coordinates
	      !during the scattering. Including it can result in rare
	      !conditions where a particle gets stuck in the ghost zones. 
	      !This can be improved. 

	      !unwind x location of particle
	      x0=p(n1)%x-p(n1)%u/gamma*c
	      y0=p(n1)%y !-p(n1)%v/gamma*c
	      z0=p(n1)%z !-p(n1)%w/gamma*c

	      !unwind wall location
              betawall2 =  -beta*1.!*max(beta, 0.99999)
	      wall2loc0=wall2loc-betawall2*c
              gammawall2=1./sqrt(1.-betawall2**2)
	      
	      !where did they meet?
	      !tfrac=abs((x0-wall2loc0)/(betawall2*c-p(n1)%u/gamma*c))
              ! changed Feb. 15, 2014
              tfrac=min(abs((x0-wall2loc0)/max(abs(betawall2*c-p(n1)%u/gamma*c),1e-9)),1.)
	      xcolis=x0+p(n1)%u/gamma*c*tfrac
	      ycolis=y0 !+p(n1)%v/gamma*c*tfrac
	      zcolis=z0 !+p(n1)%w/gamma*c*tfrac

	      !deposit current upto intersection
	      q=p(n1)%ch*real(splitratio)**(1.-real(p(n1)%splitlev))*q0
	      call zigzag(xcolis,ycolis,zcolis,x0,y0,z0,in)
              
	      !reset particle momentum, getting a kick from the wal
              p(n1)%u=gammawall2**2*gamma*(2*betawall2 - p(n1)%u/gamma*(1 + betawall2**2))
              
	      !p(n1)%u= - p(n1)%u
	      gamma=sqrt(1+(p(n1)%u**2+p(n1)%v**2+p(n1)%w**2))
	      
	      tfrac=min(abs((p(n1)%x-xcolis)/max(abs(p(n1)%x-x0),1e-9)),1.)
              !move particle from the wall position with the new velocity
 
	      p(n1)%x = xcolis + p(n1)%u/gamma*c * tfrac
	      p(n1)%y = ycolis !+ p(n1)%v/gamma*c * tfrac
	      p(n1)%z = zcolis !+ p(n1)%w/gamma*c * tfrac

	     
!now clean up the piece of trajectory behind the wall, that deposit_particles will be adding when it 
!unwinds the position of the particle by the full timestep. 
	      
	      q=-q
	      call zigzag(xcolis,ycolis,zcolis,p(n1)%x-p(n1)%u/gamma*c, & 
	              p(n1)%y-p(n1)%v/gamma*c,p(n1)%z-p(n1)%w/gamma*c,in)
	      
           endif

	enddo
	enddo


	endif

 if (leftclean .eq. 1 .and. lap .ge. lstart_jump) then
    if (lap .eq. lstart_jump) then 
       leftcleanoffset=leftcleanoffset+int(u_sh_left*c*lstep_first)
       x1in=walloc+int(u_sh_left*c*lstep_first)
    endif
    if (lap .gt. lstart_jump .and. modulo(lap, lstep_jump) .eq. 0) then
       leftcleanoffset=leftcleanoffset+int(u_sh_left*c*lstep_jump)
       x1in=walloc+int(u_sh_left*c*lstep_jump)
    endif 
 endif

	end subroutine particle_bc_user
	

	
#ifdef twoD
end module m_user
#else
end module m_user_3d
#endif

