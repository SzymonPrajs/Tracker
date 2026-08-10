# Register census appendix

This appendix preserves the register-count evidence from the pinned sources. It is an inventory for navigation and audit, not a replacement for register semantics in the TRM or ESP-IDF drivers.

## HP processor-complex register schemas

TRM v0.6 numbers the following 137 schemas. Several are arrays or parameterized families, so this is not the number of individually addressable instances.

```text
Register 2.1. mvendorid
Register 2.2. marchid
Register 2.3. mimpid
Register 2.4. mhartid
Register 2.5. mstatus
Register 2.6. misa
Register 2.7. mideleg
Register 2.8. mie
Register 2.9. mtvec
Register 2.10. mtvt
Register 2.11. mscratch
Register 2.12. mepc
Register 2.13. mcause
Register 2.14. mtval
Register 2.15. mip
Register 2.16. mnxti
Register 2.17. mintstatus
Register 2.18. mintthresh
Register 2.19. mscratchcsw
Register 2.20. mscratchcswl
Register 2.21. mclicbase
Register 2.22. ustatus
Register 2.23. utvec
Register 2.24. utvt
Register 2.25. uscratch
Register 2.26. uepc
Register 2.27. ucause
Register 2.28. unxti
Register 2.29. uintthresh
Register 2.30. uclicbase
Register 2.31. uintstatus
Register 2.32. mcounteren
Register 2.33. mcountinhibit
Register 2.34. mhpmevent8
Register 2.35. mhpmevent9
Register 2.36. mhpmevent13
Register 2.37. mcycle
Register 2.38. minstret
Register 2.39. mhpmcounter8
Register 2.40. mhpmcounter9
Register 2.41. mhpmcounter13
Register 2.42. mcycleh
Register 2.43. minstreth
Register 2.44. mhpmcounter8h
Register 2.45. mhpmcounter9h
Register 2.46. mhpmcounter13h
Register 2.47. fflags
Register 2.48. frm
Register 2.49. fcsr
Register 2.50. fxsr
Register 2.51. cycle
Register 2.52. time
Register 2.53. instret
Register 2.54. hpmcounter8
Register 2.55. hpmcounter9
Register 2.56. hpmcounter13
Register 2.57. cycleh
Register 2.58. timeh
Register 2.59. instreth
Register 2.60. hpmcounter8h
Register 2.61. hpmcounter9h
Register 2.62. hpmcounter13h
Register 2.63. mext_ill
Register 2.64. mext_hwlp_status
Register 2.65. mext_pie_status
Register 2.66. jvt
Register 2.67. mexstatus
Register 2.68. mhint
Register 2.69. ldpc0
Register 2.70. ldpc1
Register 2.71. ldtval0
Register 2.72. ldtval1
Register 2.73. stpc0
Register 2.74. stpc1
Register 2.75. stpc2
Register 2.76. sttval0
Register 2.77. sttval1
Register 2.78. sttval2
Register 2.79. mhwloop0_start_addr
Register 2.80. mhwloop0_end_addr
Register 2.81. mhwloop0_count
Register 2.82. mhwloop1_start_addr
Register 2.83. mhwloop1_end_addr
Register 2.84. mhwloop1_count
Register 2.85. mext_ill_reg
Register 2.86. mhwloop_state_reg
Register 2.87. uhwloop0_start_addr
Register 2.88. uhwloop0_end_addr
Register 2.89. uhwloop0_count
Register 2.90. uhwloop1_start_addr
Register 2.91. uhwloop1_end_addr
Register 2.92. uhwloop1_count
Register 2.93. uhwloop_state_reg
Register 2.94. mcliccfg
Register 2.95. clicinfo
Register 2.96. clicintip
Register 2.97. clicintie
Register 2.98. clicintattr
Register 2.99. clicintctl
Register 2.100. msip
Register 2.101. mtimecmplo
Register 2.102. mtimecmphi
Register 2.103. mtimeloadlo
Register 2.104. mtimeloadhi
Register 2.105. mtimectl
Register 2.106. mtimelo
Register 2.107. mtimehi
Register 2.108. pmpcfg0
Register 2.109. pmpcfg1
Register 2.110. pmpcfg2
Register 2.111. pmpcfg3
Register 2.112. pmpcfg4
Register 2.113. pmpcfg5
Register 2.114. pmpcfg6
Register 2.115. pmpcfg7
Register 2.116. pmpXcfg
Register 2.117. pmpaddrn
Register 2.118. pma_cfgn
Register 2.119. pma_addrn
Register 2.120. dcsr
Register 2.121. dpc
Register 2.122. dscratch0
Register 2.123. dscratch1
Register 2.124. dmcs2
Register 2.125. tselect
Register 2.126. tdata1
Register 2.127. tdata2
Register 2.128. tdata3
Register 2.129. tinfo
Register 2.130. tcontrol
Register 2.131. MCONTEXT
Register 2.132. mcontrol
Register 2.133. maddress
Register 2.134. mhcr
Register 2.135. cpu_gpio_oen
Register 2.136. cpu_gpio_in
Register 2.137. cpu_gpio_out
```

## SoC MMIO definitions by generated header

The ESP-IDF v6.0.2 ESP32-P4 v3 register tree has 104 `*_reg.h` files. The counting rule found 5,936 named definitions in the following 102 files; `io_mux_reg.h` and `wdev_reg.h` contain no declarations matching that rule.

```text
count  header
 380  gpio_reg.h
 271  h264_dma_reg.h
 247  efuse_reg.h
 245  cache_reg.h
 188  ahb_dma_reg.h
 187  dma2d_reg.h
 183  axi_dma_reg.h
 157  isp_reg.h
 145  interrupt_core0_reg.h
 145  interrupt_core1_reg.h
 141  dw_gdma_reg.h
 138  soc_etm_reg.h
 137  pmu_reg.h
 135  pmu_eco5_reg.h
 134  pvt_reg.h
 130  dma_pms_eco5_reg.h
 128  dma_pms_reg.h
 125  emac_reg.h
 116  axi_perf_mon_reg.h
 105  hp_system_reg.h
  87  spi_mem_s_reg.h
  84  spi_mem_c_reg.h
  83  mcpwm_reg.h
  81  lp_analog_peri_reg.h
  74  mipi_dsi_host_reg.h
  73  ledc_reg.h
  73  lp_system_reg.h
  70  h264_reg.h
  66  i3c_mst_mem_reg.h
  64  assist_debug_reg.h
  62  hp_sys_clkrst_reg.h
  59  lp_gpio_reg.h
  52  rmt_reg.h
  44  sdmmc_reg.h
  43  lp_spi_reg.h
  43  spi1_mem_c_reg.h
  43  spi1_mem_s_reg.h
  42  mipi_csi_host_reg.h
  41  jpeg_reg.h
  40  gpio_ext_reg.h
  38  lp_i2s_reg.h
  38  mipi_dsi_bridge_reg.h
  38  spi_reg.h
  38  uart_reg.h
  37  ppa_reg.h
  36  i3c_mst_reg.h
  36  systimer_reg.h
  35  i2c_reg.h
  35  timer_group_reg.h
  35  twai_reg.h
  35  usb_serial_jtag_reg.h
  34  aes_reg.h
  34  lp_i2c_reg.h
  33  lp_uart_reg.h
  33  uhci_reg.h
  30  aes_eco5_reg.h
  30  pcnt_reg.h
  28  adc_eco5_reg.h
  28  adc_reg.h
  27  iomux_mspi_pin_reg.h
  25  lp_mailbox_reg.h
  25  touch_reg.h
  24  i2s_reg.h
  22  parl_io_reg.h
  21  lp_adc_reg.h
  21  lp_clkrst_reg.h
  21  lp_iomux_reg.h
  21  lp_peri_pms_eco5_reg.h
  21  rtclockcali_reg.h
  19  hp_peri_pms_eco5_reg.h
  19  icm_sys_reg.h
  19  lcd_cam_reg.h
  19  lp_timer_reg.h
  19  trace_reg.h
  18  hp_peri_pms_reg.h
  17  hmac_reg.h
  17  pau_reg.h
  16  bitscrambler_reg.h
  16  i3c_slv_reg.h
  16  mipi_csi_bridge_reg.h
  15  i2c_ana_mst_reg.h
  14  ecdsa_reg.h
  14  keymng_reg.h
  14  lp_wdt_reg.h
  13  mem_monitor_reg.h
  13  rsa_reg.h
  12  sha_reg.h
  11  lpperi_reg.h
  11  sha_eco5_reg.h
  10  huk_reg.h
  10  tsens_reg.h
   8  ds_reg.h
   8  lp_peri_pms_reg.h
   7  hp2lp_peri_pms_eco5_reg.h
   7  hp2lp_peri_pms_reg.h
   6  ecc_mult_reg.h
   6  lp2hp_peri_pms_eco5_reg.h
   6  lp2hp_peri_pms_reg.h
   6  lp_intr_reg.h
   4  icm_sys_qos_reg.h
   4  trng_reg.h
   2  usb_wrap_reg.h
```

The count includes repeated channels, instances, indexed registers, aliases, and ECO overlay headers. Never sum it with the 137 TRM processor schemas: the scopes and counting methods differ.
