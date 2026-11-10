// Required so R can find the Rust-registered routines.
void R_init_emeansR_extendr(void *dll);

void R_init_emeansR(void *dll) {
    R_init_emeansR_extendr(dll);
}
