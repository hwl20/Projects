package DcaCalculator;

import java.util.Scanner;

public class dca {
    public static void tryout() {
        Scanner scanner = new Scanner(System.in);

        System.out.print("Type in amount put in monthly: ");
        double yearlyAmount = scanner.nextInt() * 12;

        System.out.print("Time horizon: ");
        double year = scanner.nextInt();

        System.out.print("Interest per annum (%): ");
        double interest = scanner.nextDouble()/100;

        double amount = ((1+interest)*yearlyAmount*(Math.pow((1+interest),year)-1))/(interest);
        amount = amount/(1+interest);
//        double amount = 0;
//        for (int i = 1; i<=year; i++){
//            amount = amount*(1+interest) + yearlyAmount;
//        }
        System.out.printf("Amount at the end of %.0f years is: %.2f", year, amount);
    }
}
