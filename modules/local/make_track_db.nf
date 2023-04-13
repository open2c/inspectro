#!/usr/bin/env nextflow

nextflow.enable.dsl=2

process make_track_db {
    
    label 'process_medium'
    container = 'tariship/inspectrov1@sha256:90ebe2ee28a5b6bf7aa67e2abfdfd82f93dd39e02ce37f9bdbdf1e12ba1db3e6'
    publishDir "${params.outdir}"
    
    input:
        path config
        path bins
        path meta

    output:
        path "*.h5", emit: trackdb

    script:

    """ 
    make_track_db.py --config ${config} --bins ${bins} --meta ${meta}
    """ 
}
