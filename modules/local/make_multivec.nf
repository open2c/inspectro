#!/usr/bin/env nextflow

nextflow.enable.dsl=2

process make_multivec {

    label 'process_single'
    container = 'tariship/inspectrov1@sha256:90ebe2ee28a5b6bf7aa67e2abfdfd82f93dd39e02ce37f9bdbdf1e12ba1db3e6'
    publishDir "${params.outdir}"

    input:
        path config
        path eigvals
        path eigvecs

    output:
        path "*.mv5"

    script:

    """ 
    make_multivec.py --config ${config} --eigvals ${eigvals} --eigvecs ${eigvecs} 
    """ 
}